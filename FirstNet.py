import torch
import torch.nn.functional as F
import numpy as np


class FirstNet(object):
    def __init__(self, batch_n, input_data_size, output_data_size, lr, inputInfo, cuda, device_n=6):
        self.batch_n = batch_n
        self.input_data_size = input_data_size
        self.output_data_size = output_data_size
        self.lr = lr
        self.inputInfo = inputInfo
        self.cuda = cuda
        self.device_n = device_n

        self.models = torch.nn.Sequential(
            torch.nn.Linear(self.input_data_size, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 160),
            torch.nn.ReLU(),
            torch.nn.Linear(160, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, self.output_data_size)
        ).to(self.cuda)

        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=self.lr)

    def doActions(self):
        batch_action = []
        model_output = self.models(torch.Tensor(self.inputInfo).to(self.cuda))
        for batch in range(self.batch_n):
            actions = []
            action_probs = F.softmax(model_output[batch].cpu(), dim=-1).detach().numpy()
            action = np.random.choice(self.output_data_size, p=action_probs)
            actions.append(action)
            batch_action.extend(actions)

        countStageNum = 0
        for i in range(self.batch_n):
            countStageNum += batch_action[i]
        if batch_action[self.batch_n - 1] == 0:
            batch_action[self.batch_n - 1] = 1
            if countStageNum == self.device_n:
                i = self.batch_n - 2
                while batch_action[i] == 0:
                    i -= 1
                batch_action[i] -= 1

        return model_output, batch_action

    def doGreedyActions(self):
        batch_action = []
        model_output = self.models(torch.Tensor(self.inputInfo).to(self.cuda))
        for batch in range(self.batch_n):
            actions = []
            action = F.softmax(model_output[batch], dim=-1).argmax(dim=-1).cpu().numpy()
            actions.append(action)
            batch_action.extend(actions)

        countStageNum = 0
        for i in range(self.batch_n):
            countStageNum += batch_action[i]
        if batch_action[self.batch_n - 1] == 0:
            batch_action[self.batch_n - 1] = 1
            if countStageNum == self.device_n:
                i = self.batch_n - 1
                while batch_action[i] == 0:
                    i -= 1
                batch_action[i] -= 1

        return batch_action

    def showFNProbability(self):
        model_output = self.models(torch.Tensor(self.inputInfo).to(self.cuda))
        print("FN_Probability: ", F.softmax(model_output, dim=-1))

    def countAction(self, batch_action, batch_n):
        stageNumber = 0
        for batchTime in range(batch_n):
            stageNumber += batch_action[batchTime]
        return stageNumber

    def saveFN_All(self):
        torch.save(self.models, './pth/FN_All.pth')

    def saveFN_Parameters(self):
        torch.save(self.models.state_dict(), './pth/FN_Parameters.pth')

    def loadFN_All(self):
        self.models = torch.load('./pth/FN_All.pth')

    def loadFN_Parameters(self):
        self.models.load_state_dict(torch.load('./pth/FN_Parameters.pth'))

    def parametersIsNAN(self):
        model_output = self.models(torch.Tensor(self.inputInfo).to(self.cuda))
        if np.isnan(model_output[0][0].cpu().detach().numpy()):
            return 1
        return 0
