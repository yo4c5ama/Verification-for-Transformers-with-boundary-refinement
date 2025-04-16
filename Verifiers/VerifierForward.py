# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import math, time, random, copy
from Verifiers import Verifier
from Verifiers.Bounds import Bounds
from Verifiers.utils import check
import numpy as np

# can only accept one example in each batch
class VerifierForward(Verifier):
    def __init__(self, args, target, logger):
        super(VerifierForward, self).__init__(args, target, logger)
        self.ibp = args.method == "ibp"
    def verify_safety(self, example, embeddings, index, eps):

        if self.double_z:
            l, u, bounds = self.get_l_u(example, embeddings, index, eps)
            # self.args.double_z = False
            # l_2, u_2, bounds_2 = self.get_l_u(example, embeddings, index, eps)
            #
            # lw_1 = torch.norm(bounds_1.lw, p=1. / (1. - 1. / self.p) if self.p != 1 else float("inf"), dim=-2)[0][0][0]
            # lw_2 = torch.norm(bounds_2.lw, p=1. / (1. - 1. / self.p) if self.p != 1 else float("inf"), dim=-2)[0][0][0]
            # lb_1 = bounds_1.lb[0][0][0]
            # lb_2 = bounds_2.lb[0][0][0]
            # uw_1 = torch.norm(bounds_1.uw, p=1. / (1. - 1. / self.p) if self.p != 1 else float("inf"), dim=-2)[0][0][0]
            # uw_2 = torch.norm(bounds_2.uw, p=1. / (1. - 1. / self.p) if self.p != 1 else float("inf"), dim=-2)[0][0][0]
            # ub_1 = bounds_1.ub[0][0][0]
            # ub_2 = bounds_2.ub[0][0][0]
            # u_test = (ub_2 * uw_1 - ub_1 * uw_2)/(uw_1 - uw_2)
            # u_x = (ub_2 - ub_1)/(uw_1 - uw_2)
            # l_test = (lb_2 * lw_1 - lb_1 * lw_2)/(lw_1 - lw_2)
            # l_x = (lb_2 - lb_1)/(lw_1 - lw_2)
            # l = max(l_1, l_2, l_test)
            # u = min(u_1, u_2, u_test)
            # # l = max(l_1, l_2)
            # # u = min(u_1, u_2)
            # if l > u:
            #     print("\n",l,u,"error!!!!!!","\n")
            self.args.double_z = True
            if example["label"] == 0:
                safe = l > 0
            else:
                safe = u < 0

            if self.verbose:
                print("Safe" if safe else "Unsafe")
            return safe
        else:
            l, u = self.get_l_u(example, embeddings, index, eps)
            if example["label"] == 0:
                safe = l > 0
            else:
                safe = u < 0

            if self.verbose:
                print("Safe" if safe else "Unsafe")
            return safe

    def get_l_u(self, example, embeddings, index, eps):
        errorType = OSError if self.debug else AssertionError

        try:
            with torch.no_grad():
                bounds = self._bound_input(embeddings, index=index, eps=eps) # hard-coded yet

                check("embedding", bounds=bounds, std=self.std["embedding_output"], verbose=self.verbose)
            
                if self.verbose:
                    bounds.print("embedding")

                for i, layer in enumerate(self.encoding_layers):
                    attention_scores, attention_probs, bounds = self._bound_layer(bounds, layer)
                    check("layer %d attention_scores" % i, 
                        bounds=attention_scores, std=self.std["attention_scores"][i][0], verbose=self.verbose)
                    check("layer %d attention_probs" % i, 
                        bounds=attention_probs, std=self.std["attention_probs"][i][0], verbose=self.verbose)
                    check("layer %d" % i, bounds=bounds, std=self.std["encoded_layers"][i], verbose=self.verbose)

                    
                bounds = self._bound_pooling(bounds, self.pooler)
                check("pooled output", bounds=bounds, std=self.std["pooled_output"], verbose=self.verbose)

                # safety = self._bound_classifier(bounds, self.classifier, example["label"])
                # return safety
                l, u, bounds = self._get_final_inteval(bounds, self.classifier, example["label"])
            return l,u, bounds

        except errorType as err: # for debug
            if self.verbose:
                print("Warning: failed assertion")
                print(err)
            print("Warning: failed assertion", eps)
            return False

    def _bound_input(self, embeddings, index, eps):
        length, dim = embeddings.shape[1], embeddings.shape[2]

        w = torch.zeros((length, dim * self.perturbed_words, dim)).to(self.device)
        b = embeddings[0]   
        lb, ub = b, b.clone()

        if self.perturbed_words == 1:
            if self.ibp:
                lb[index], ub[index] = lb[index] - eps, ub[index] + eps
            else:
                w[index] = torch.eye(dim).to(self.device)
        else:
            if self.ibp:
                for i in range(self.perturbed_words):
                    lb[index[i]], ub[index[i]] = lb[index[i]] - eps, ub[index[i]] + eps
            else:
                for i in range(self.perturbed_words):
                    w[index[i], (dim * i):(dim * (i + 1)), :] = torch.eye(dim).to(self.device)
            
        lw = w.unsqueeze(0)
        uw = lw.clone()
        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)

        bounds = Bounds(self.args, self.p, eps, lw=lw, lb=lb, uw=uw, ub=ub)

        bounds = bounds.layer_norm(self.embeddings.LayerNorm, self.layer_norm)

        return bounds

    def _bound_layer(self, bounds_input, layer):
        start_time = time.time()

        # main self-attention
        attention1, attention2, attention3, attention4 = \
            self._bound_attention(bounds_input, layer.attention)

        def add_norm(attention):
            attention = attention.dense(layer.attention.output.dense)
            attention = attention.add(bounds_input).layer_norm(layer.attention.output.LayerNorm, self.layer_norm)
            if self.verbose:
                attention.print("after attention layernorm")
                attention.dense(layer.intermediate.dense).print("intermediate pre-activation")
                print("dense norm", torch.norm(layer.intermediate.dense.weight, p=self.p))
                start_time = time.time()

            intermediate = attention.dense(layer.intermediate.dense).act(self.hidden_act)

            if self.verbose:
                intermediate.print("intermediate")
                start_time = time.time()

            dense = intermediate.dense(layer.output.dense).add(attention)
            del(intermediate)
            del(attention)

            if self.verbose:
                print("dense norm", torch.norm(layer.output.dense.weight, p=self.p))
                dense.print("output pre layer norm")

            output = dense.layer_norm(layer.output.LayerNorm, self.layer_norm)
            return output

        output1 = add_norm(attention1)
        output2 = add_norm(attention2)
        output3 = add_norm(attention3)
        output4 = add_norm(attention4)
        del (bounds_input)

        if self.verbose:
            output1.print("output")
            # print(" time", time.time() - start_time)
            start_time = time.time()            

        return output1, output2, output3, output4



    def _bound_attention(self, bounds_input, attention):
        num_attention_heads = attention.self.num_attention_heads
        attention_head_size = attention.self.attention_head_size

        query = bounds_input.dense(attention.self.query)
        key = bounds_input.dense(attention.self.key)
        value = bounds_input.dense(attention.self.value)

        # del(bounds_input)

        def transpose_for_scores(x):
            def transpose_w(x):
                return x\
                    .reshape(
                        x.shape[0], x.shape[1], x.shape[2], 
                        num_attention_heads, attention_head_size)\
                    .permute(0, 3, 1, 2, 4)\
                    .reshape(-1, x.shape[1], x.shape[2], attention_head_size)
            def transpose_b(x):
                return x\
                    .reshape(
                        x.shape[0], x.shape[1], num_attention_heads, attention_head_size)\
                    .permute(0, 2, 1, 3)\
                    .reshape(-1, x.shape[1], attention_head_size)
            x.lw = transpose_w(x.lw)
            x.uw = transpose_w(x.uw)
            x.lb = transpose_b(x.lb)
            x.ub = transpose_b(x.ub)
            x.update_shape()

        transpose_for_scores(query)
        transpose_for_scores(key)
        # transpose_for_scores(bounds_input)
        # l_a, u_a = query.concretize()
        # l_b, u_b = key.concretize()
        # TODO: no attention mask for now (doesn't matter for batch_size=1)
        qk1, qk2 = query.dot_product(key, verbose=self.verbose)
        # u_max = u_a @ u_b.transpose(-2, -1)
        # l_max = l_a @ l_b.transpose(-2, -1)


        # def get_cross_point(bounds_input, Bounds1, Bounds2):
        #
        #     def get_w_b(w0, w1, b0, b1):
        #         w = torch.linalg.pinv(w0) @ w1
        #         # error_pinv = torch.norm(w0 @ w - w1)/torch.norm(w1)
        #         b = b1 - (b0.unsqueeze(-2) @ w).squeeze(-2)
        #         return w, b
        #
        #     Bounds1_l_w, Bounds1_l_b = get_w_b(bounds_input.lw, Bounds1.lw, bounds_input.lb, Bounds1.lb)
        #     Bounds1_u_w, Bounds1_u_b = get_w_b(bounds_input.uw, Bounds1.uw, bounds_input.ub, Bounds1.ub)
        #     Bounds2_l_w, Bounds2_l_b = get_w_b(bounds_input.lw, Bounds2.lw, bounds_input.lb, Bounds2.lb)
        #     Bounds2_u_w, Bounds2_u_b = get_w_b(bounds_input.uw, Bounds2.uw, bounds_input.ub, Bounds2.ub)
        #
        #     l_cross_point_x = ((Bounds2_l_b - Bounds1_l_b).unsqueeze(-2) \
        #                       @ torch.linalg.pinv(Bounds1_l_w-Bounds2_l_w)).squeeze(-2)
        #
        #     # l_input,u_input = bounds_input.concretize()
        #     l_cross_point_y_1 = (l_cross_point_x.unsqueeze(-2) @ Bounds1_l_w).squeeze(-2) + Bounds1_l_b
        #     l_cross_point_y_2 = (l_cross_point_x.unsqueeze(-2) @ Bounds2_l_w).squeeze(-2) + Bounds2_l_b
        #     l_cross_point_y = (l_cross_point_y_1 + l_cross_point_y_2)/2
        #     # error = torch.norm(l_cross_point_y_2-l_cross_point_y_1)/torch.norm(l_cross_point_y_1)
        #     u_cross_point_x = ((Bounds2_u_b - Bounds1_u_b).unsqueeze(-2) \
        #                        @ torch.linalg.pinv(Bounds1_u_w - Bounds2_u_w)).squeeze(-2)
        #     u_cross_point_y_1 = (u_cross_point_x.unsqueeze(-2) @ Bounds1_u_w).squeeze(-2) + Bounds1_u_b
        #     u_cross_point_y_2 = (u_cross_point_x.unsqueeze(-2) @ Bounds2_u_w).squeeze(-2) + Bounds2_u_b
        #     u_cross_point_y = (u_cross_point_y_1 + u_cross_point_y_2) / 2
        #
        #     l_cross_point_x_global = ((Bounds2.lb - Bounds1.lb).unsqueeze(-2) \
        #                       @ torch.linalg.pinv(Bounds1.lw-Bounds2.lw)).squeeze(-2)
        #     l_cross_point_y_global_1 = (l_cross_point_x_global.unsqueeze(-2) @ Bounds1.lw).squeeze(-2) + Bounds1.lb
        #     l_cross_point_y_global_2 = (l_cross_point_x_global.unsqueeze(-2) @ Bounds2.lw).squeeze(-2) + Bounds2.lb
        #     l_cross_point_y_global = (l_cross_point_y_global_1 + l_cross_point_y_global_2)/2
        #     # error = torch.norm(l_cross_point_y_global_2 - l_cross_point_y_global_1) / torch.norm(l_cross_point_y_global_1)
        #     u_cross_point_x_global = ((Bounds2.ub - Bounds1.ub).unsqueeze(-2) \
        #                       @ torch.linalg.pinv(Bounds1.uw-Bounds2.uw)).squeeze(-2)
        #     u_cross_point_y_global_1 = (u_cross_point_x_global.unsqueeze(-2) @ Bounds1.uw).squeeze(-2) + Bounds1.ub
        #     u_cross_point_y_global_2 = (u_cross_point_x_global.unsqueeze(-2) @ Bounds2.uw).squeeze(-2) + Bounds2.ub
        #     u_cross_point_y_global = (u_cross_point_y_global_1 + u_cross_point_y_global_2)/2
        #     # error = torch.norm(l_cross_point_y_global_2 - l_cross_point_y_global_1) / torch.norm(l_cross_point_y_global_1)
        #     return l_cross_point_x,l_cross_point_y, u_cross_point_x, u_cross_point_y,l_cross_point_y_global,u_cross_point_y_global

        def get_cross_point(Bounds1, Bounds2):
            l_cross_point_x = ((Bounds2.lb - Bounds1.lb).unsqueeze(-2) \
                              @ torch.linalg.pinv(Bounds1.lw-Bounds2.lw)).squeeze(-2)
            l_cross_point_y_1 = (l_cross_point_x.unsqueeze(-2) @ Bounds1.lw).squeeze(-2) + Bounds1.lb
            l_cross_point_y_2 = (l_cross_point_x.unsqueeze(-2) @ Bounds2.lw).squeeze(-2) + Bounds2.lb
            l_cross_point_y= (l_cross_point_y_1 + l_cross_point_y_2)/2
            # error = torch.norm(l_cross_point_y_global_2 - l_cross_point_y_global_1) / torch.norm(l_cross_point_y_global_1)
            u_cross_point_x= ((Bounds2.ub - Bounds1.ub).unsqueeze(-2) \
                              @ torch.linalg.pinv(Bounds1.uw-Bounds2.uw)).squeeze(-2)
            u_cross_point_y_1 = (u_cross_point_x.unsqueeze(-2) @ Bounds1.uw).squeeze(-2) + Bounds1.ub
            u_cross_point_y_2 = (u_cross_point_x.unsqueeze(-2) @ Bounds2.uw).squeeze(-2) + Bounds2.ub
            u_cross_point_y = (u_cross_point_y_1 + u_cross_point_y_2)/2
            # error = torch.norm(l_cross_point_y_global_2 - l_cross_point_y_global_1) / torch.norm(l_cross_point_y_global_1)
            l1_l, l1_u, u1_l, u1_u= Bounds1.attention_bound_concretize()
            l2_l, l2_u, u2_l, u2_u = Bounds2.attention_bound_concretize()
            # a = Bounds1.eps * torch.norm(Bounds1.uw, p=float("inf"), dim=-2)
            # b = (a.unsqueeze(-2) @ torch.linalg.pinv(Bounds1.uw)).squeeze(-2)
            Bounds1.lc = l_cross_point_y
            Bounds1.uc = u_cross_point_y
            Bounds2.lc = l_cross_point_y
            Bounds2.uc = u_cross_point_y


        # l_cross_point_x,l_cross_point_y, u_cross_point_x, u_cross_point_y,l_cross_point_y_global,u_cross_point_y_global = get_cross_point(bounds_input, qk1, qk2)
        get_cross_point(qk1, qk2)

        attention_scores1 = qk1.multiply(1. / math.sqrt(attention_head_size))
        attention_scores2 = qk2.multiply(1. / math.sqrt(attention_head_size))

        # check("layer attention_scores",
        #       bounds=attention_scores1, std=self.std["attention_scores"][0][0], verbose=self.verbose)
        # check("layer attention_scores",
        #       bounds=attention_scores2, std=self.std["attention_scores"][0][0], verbose=self.verbose)
        check("test", l=attention_scores1.lc, u=attention_scores1.uc, std=self.std["attention_scores"][0][0],
              verbose=self.verbose)
        # check("test", l=l_max * (1. / math.sqrt(attention_head_size)), u=u_max * (1. / math.sqrt(attention_head_size)), std=self.std["attention_scores"][0][0],
        #       verbose=self.verbose)
        del (bounds_input)
        if self.verbose:
            attention_scores1.print("attention score")
            attention_scores2.print("attention score")

        del(query)
        del(key)
        attention_probs1 = attention_scores1.softmax(verbose=self.verbose)
        attention_probs2 = attention_scores2.softmax(verbose=self.verbose)
        check("test", l=attention_probs1.lc, u=attention_probs1.uc, std=self.std["attention_probs"][0][0],
              verbose=self.verbose)

        if self.verbose:
            attention_probs1.print("attention probs")
            attention_probs2.print("attention probs")

        transpose_for_scores(value)  

        context1, context2 = attention_probs1.context(value)
        context3, context4 = attention_probs2.context(value)
        get_cross_point(context1, context2)
        get_cross_point(context3, context4)
        if self.verbose:
            value.print("value")        
            context1.print("context")

        def transpose_back(x):
            def transpose_w(x):
                return x.permute(1, 2, 0, 3).reshape(1, x.shape[1], x.shape[2], -1)
            def transpose_b(x):
                return x.permute(1, 0, 2).reshape(1, x.shape[1], -1)

            x.lw = transpose_w(x.lw)
            x.uw = transpose_w(x.uw)
            x.lb = transpose_b(x.lb)
            x.ub = transpose_b(x.ub)
            x.lc = transpose_b(x.lc)
            x.uc = transpose_b(x.uc)
            x.update_shape()
        
        transpose_back(context1)
        transpose_back(context2)
        transpose_back(context3)
        transpose_back(context4)
        l,u = context1.concretize()
        return context1, context2, context3, context4
        
    def _bound_pooling(self, bounds, pooler):
        bounds = Bounds(
            self.args, bounds.p, bounds.eps,
            lw = bounds.lw[:, :1, :, :], lb = bounds.lb[:, :1, :],
            uw = bounds.uw[:, :1, :, :], ub = bounds.ub[:, :1, :]
        )
        if self.verbose:
            bounds.print("pooling before dense")

        bounds = bounds.dense(pooler.dense)

        if self.verbose:
            bounds.print("pooling pre-activation")

        bounds = bounds.tanh()

        if self.verbose:
            bounds.print("pooling after activation")
        return bounds

    def _bound_classifier(self, bounds, classifier, label):
        classifier = copy.deepcopy(classifier)
        classifier.weight[0, :] -= classifier.weight[1, :]
        classifier.bias[0] -= classifier.bias[1]

        if self.verbose:
            bounds.print("before dense")
            print(torch.norm(classifier.weight[0, :]))
            print(torch.mean(torch.norm(bounds.lw, dim=-2)))
            print(torch.mean(torch.norm(bounds.dense(classifier).lw, dim=-2)))

        bounds = bounds.dense(classifier)
        
        if self.verbose:
            bounds.print("after dense")

        l, u = bounds.concretize()

        if self.verbose:
            print(l[0][0][0])
            print(u[0][0][0])

        if label == 0:
            safe = l[0][0][0] > 0
        else:
            safe = u[0][0][0] < 0

        if self.verbose:
            print("Safe" if safe else "Unsafe")

        return safe

    def _get_final_inteval(self, bounds, classifier, label):
        classifier = copy.deepcopy(classifier)
        classifier.weight[0, :] -= classifier.weight[1, :]
        classifier.bias[0] -= classifier.bias[1]

        if self.verbose:
            bounds.print("before dense")
            print(torch.norm(classifier.weight[0, :]))
            print(torch.mean(torch.norm(bounds.lw, dim=-2)))
            print(torch.mean(torch.norm(bounds.dense(classifier).lw, dim=-2)))

        bounds = bounds.dense(classifier)

        if self.verbose:
            bounds.print("after dense")

        l, u = bounds.concretize()
        return l[0][0][0], u[0][0][0], bounds
