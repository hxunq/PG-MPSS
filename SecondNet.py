from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from InputInfo.InceptionV3Info import InceptionV3Info as ModelInfo


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, batch_n, cuda, hidden_size=20):
        super().__init__()
        self.cuda = cuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers).to(self.cuda)
        self.fc = nn.Linear(hidden_size, output_size).to(self.cuda)
        self.batch_n = batch_n

    def forward(self, _x):
        x = _x.view(self.batch_n, 1, -1)
        x, _ = self.lstm(x)
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.fc(x)
        return x


class testNet(nn.Module):
    def __init__(self, input_size, output_size, cuda):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.cuda = cuda

        self.models = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 96),
            torch.nn.ReLU(),
            torch.nn.Linear(96, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, self.output_size)
        ).to(self.cuda)

    def forward(self, x):
        return self.models(x)


class SecondNet(object):
    def __init__(self, input_data_size, output_data_size, lr, DNNInfo, batch_n, cuda):
        self.batch_n = batch_n
        self.input_data_size = input_data_size
        self.output_data_size = output_data_size
        self.lr = lr
        self.DNNInfo = DNNInfo
        self.cuda = cuda
        self.model = LSTM(input_size=self.input_data_size, output_size=self.output_data_size, num_layers=2, batch_n=self.batch_n, cuda=self.cuda)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def inputInfo_pre_handle(self, FNbatch_action, layer_num, device_n):
        pre_handle_DNNInfo = []
        stageComposition = []
        stageInfo = 0
        putInSameStage_layer_num = 0
        for layer_i in range(layer_num):
            if (FNbatch_action[layer_i] == 0):
                putInSameStage_layer_num += 1
            else:
                stageComposition.append(putInSameStage_layer_num + 1)
                stageInfo = copy.deepcopy(self.DNNInfo[layer_i])
                while putInSameStage_layer_num > 0:
                    stageInfo = self.merge(self.DNNInfo[layer_i - putInSameStage_layer_num], stageInfo)
                    putInSameStage_layer_num -= 1
                pre_handle_DNNInfo.append(stageInfo)

        if len(pre_handle_DNNInfo) < device_n:
            timeInfo = ModelInfo().getExeTime()
            stageTime = np.zeros(device_n)
            j = 0
            for i in range(len(stageComposition)):
                k = stageComposition[i]
                while k > 0:
                    k -= 1
                    stageTime[i] += timeInfo[0][j]
                    j += 1

            while len(pre_handle_DNNInfo) < device_n:
                long_stageTime_index = len(stageTime) - 1
                for i in range(len(stageTime - 1)):
                    if stageTime[long_stageTime_index] < stageTime[i]:
                        long_stageTime_index = i
                pre_handle_DNNInfo.append(0)
                for i in range(len(pre_handle_DNNInfo) - long_stageTime_index - 1):
                    pre_handle_DNNInfo[len(pre_handle_DNNInfo) - i - 1] = pre_handle_DNNInfo[len(pre_handle_DNNInfo) - i - 2]
                stageComposition.append(0)
                for i in range(len(pre_handle_DNNInfo) - long_stageTime_index - 1):
                    stageComposition[len(pre_handle_DNNInfo) - i - 1] = stageComposition[len(pre_handle_DNNInfo) - i - 2]
                stageComposition[long_stageTime_index + 1] = -1
                for i in range(len(pre_handle_DNNInfo) - long_stageTime_index - 1):
                    stageTime[len(pre_handle_DNNInfo) - i - 1] = stageTime[len(pre_handle_DNNInfo) - i - 2]
                stageTime[long_stageTime_index] = stageTime[long_stageTime_index] * 0.5
                stage_i = 1
                while stageComposition[long_stageTime_index + stage_i] == -1:
                    stageTime[long_stageTime_index + stage_i] = stageTime[long_stageTime_index]
                    stage_i += 1
                    if long_stageTime_index + stage_i >= len(stageComposition):
                        break

        return pre_handle_DNNInfo, stageComposition

    def doActions(self, inputInfo):
        model_output = self.model(torch.Tensor(np.array(inputInfo)).to(self.cuda))
        batch_action = []
        for batch in range(len(inputInfo)):
            actions = []
            action_probs = F.softmax(model_output[batch].cpu(), dim=-1).detach().numpy()
            action = np.random.choice(self.output_data_size, p=action_probs)
            actions.append(action)
            batch_action.extend(actions)
        return model_output, batch_action

    def doOtherActions(self, inputInfo):
        model_output = self.model(torch.Tensor(np.array(inputInfo)).to(self.cuda))
        batch_action = []
        actionToChoose = torch.Tensor(np.zeros(self.output_data_size))
        for batch in range(self.batch_n):
            actions = []
            out_batch = model_output[batch].cpu() + actionToChoose
            action_probs = F.softmax(out_batch, dim=-1).detach().numpy()
            action = np.random.choice(self.output_data_size, p=action_probs)
            actionToChoose[action] = -999
            actions.append(action)
            batch_action.extend(actions)
        return model_output, batch_action

    def doOtherGreedyActions(self, inputInfo):
        model_output = self.model(torch.Tensor(np.array(inputInfo)).to(self.cuda))
        batch_action = []
        actionToChoose = torch.Tensor(np.zeros(self.output_data_size))
        for batch in range(self.batch_n):
            actions = []
            out_batch = model_output[batch].cpu() + actionToChoose
            action = F.softmax(out_batch, dim=-1).argmax(dim=-1)
            actions.append(action)
            actionToChoose[action] = -999
            batch_action.extend(actions)
        return batch_action

    def update(self, model_output, batch_action, reward_tensor):
        action_tensor = torch.LongTensor(batch_action)
        reward_tensor = reward_tensor.to(self.cuda)
        log_probs = torch.log(F.softmax(model_output, dim=-1))
        selected_log_probs = reward_tensor * log_probs[np.arange(len(action_tensor)), action_tensor]
        loss = -selected_log_probs.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def merge(self, a, b):
        for length in range(len(a)):
            b[length] = max(a[length], b[length])
        return b

    def saveSN_All(self):
        torch.save(self.model, 'pth/SN_All.pth')

    def saveSN_Parameters(self):
        torch.save(self.model.state_dict(), 'pth/SN_Parameters.pth')

    def loadSN_All(self):
        self.model = torch.load('pth/SN_All.pth')

    def loadSN_Parameters(self):
        self.model.load_state_dict(torch.load('pth/SN_Parameters.pth'))
