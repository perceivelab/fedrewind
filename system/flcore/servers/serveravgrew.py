# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from flcore.clients.clientavgrew import clientAVGRew
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
import wandb


class FedAvgRew(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.rewind_ratio = args.rewind_ratio
        self.rewind_epochs = args.rewind_epochs
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVGRew)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def define_metrics(self):
        if self.no_wandb:
            return
        super().define_metrics()
  
        for client in self.clients:
            wandb.define_metric(f"test/model_{client.id}/acc", step_metric="round")
            wandb.define_metric(f"test/model_{client.id}/bal", step_metric="round")
            wandb.define_metric(f"train/model_{client.id}/round_train_loss_{client.id}", step_metric="round")
            wandb.define_metric(f"test/model_{client.id}/round_test_acc_{client.id}", step_metric="round")
            wandb.define_metric(f'train/model_loss_{client.train_model_id}', step_metric="round")
            wandb.define_metric(f"test/model_{client.id}/test_std", step_metric="round")
            wandb.define_metric(f"test/model_{client.id}/test_std_on_train", step_metric="round")
            if self.rewind_ratio > 0 or self.rewind_epochs > 0:
                wandb.define_metric(f"rewind/rewind_phase_loss_{client.id}", step_metric="rewind_step")
                wandb.define_metric(f"train/model_{client.id}/pre_rewind_loss_on_local", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/pre_rewind_loss_on_previous", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/post_rewind_loss_on_local", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/post_rewind_loss_on_previous", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/atend_loss_on_previous", step_metric="round")
                wandb.define_metric(f"train/model_{client.id}/atend_loss_on_local", step_metric="round")
                wandb.define_metric(f"rewind/rewind_loss_{client.id}", step_metric="round")

            for test_client in self.clients:
                wandb.define_metric(f"train/model_{client.id}/round_train_loss_{client.id}_on_{test_client.id}", step_metric="round")
                wandb.define_metric(f"test/model_{client.id}/round_test_acc_{client.id}_on_{test_client.id}", step_metric="round")

    def train_set_previous_node(self):
        if self.rewind_ratio == 0:
            return
        previous_nodes = list(range(self.num_clients))
        for client in self.clients:
            if len(previous_nodes) == 1:
                previous_node_index = previous_nodes[0]
            else:
                client_in_list = False
                if client.id in previous_nodes:
                    client_in_list = True
                if client_in_list:
                    previous_nodes.remove(client.id)
                previous_node_index = np.random.randint(0, len(previous_nodes))
                previous_node_id = previous_nodes[previous_node_index]
                previous_nodes.remove(previous_node_id)
                if client_in_list:
                    previous_nodes.append(client.id)
            client.rewind_previous_node.append(self.clients[previous_node_index])

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            if self.round > 1:
                self.train_set_previous_node()

            for client in self.selected_clients:
                client.round = self.round
                print ( f"Training client {client.id} round {client.round}" )
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVGRew)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
