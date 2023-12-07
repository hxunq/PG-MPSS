import numpy as np
import torch
import copy
from InputInfo.InceptionV3Info import InceptionV3Info as ModelInfo


def get_reward_tensor3(batch_n, batch_action):
    invalid_action_reward = np.empty(batch_n)
    devicesNumber = 0
    for i in range(batch_n):
        devicesNumber += batch_action[i]
    reward = - (devicesNumber - 4) * 10
    for i in range(batch_n):
        invalid_action_reward[i] = reward

    reward_tensor = torch.FloatTensor(invalid_action_reward)

    return reward_tensor


def get_reward_tensor4(batch_n, stageComposition, stageTime, epochTime):
    rewardCap = ModelInfo().getRewardCap()
    stageReward = rewardCap - stageTime
    layerReward = np.zeros(batch_n)
    layer_n = 0
    for stage_n in range(len(stageComposition)):
        stage_includeLayer_number = stageComposition[stage_n]
        while stage_includeLayer_number > 0:
            layerReward[layer_n] = stageReward[stage_n] - pow(epochTime, 0.5)
            layer_n += 1
            stage_includeLayer_number -= 1

    layerReward_tensor = torch.FloatTensor(layerReward)

    return layerReward_tensor


def get_reward_tensor5(stageTime, epochTime):
    reward = - stageTime
    for i in range(len(stageTime)):
        reward[i] += -pow(epochTime, 0.5)
    reward_tensor = torch.FloatTensor(reward)

    return reward_tensor


def get_reward_tensor7(stageTime, epochTime, sum_reward_SN, epoch):
    sum_reward_SN += pow(epochTime, 0.5)
    avg_reward_SN = sum_reward_SN / (epoch + 1)
    reward = np.zeros(len(stageTime))
    for i in range(len(stageTime)):
        reward[i] = - pow(epochTime, 0.5) + avg_reward_SN

    reward_tensor = torch.FloatTensor(reward)

    return reward_tensor, sum_reward_SN


def getStageTime(batch_action, stageComposition, minibatch=8):
    timeInfo = ModelInfo().getExeTime()
    stageTime = np.zeros(len(stageComposition))
    tempStageComposition = copy.deepcopy(stageComposition)

    epochTime = 0
    j = 0
    for i in range(len(stageComposition)):
        k = tempStageComposition[i]

        if tempStageComposition[i] < 0:
            e = 1
            while tempStageComposition[i - e] < 0:
                e += 1
            k = tempStageComposition[i - e]
            j -= tempStageComposition[i - e]
        while k > 0:
            k -= 1
            stageTime[i] += timeInfo[batch_action[i]][j]
            j += 1

    stage_i = 0
    while stage_i < len(stageComposition):
        if stageComposition[stage_i] < 0:
            DDP_stage_first = stage_i - 1
            while stageComposition[stage_i] < 0:
                stage_i += 1
                if stage_i >= len(stageComposition):
                    break
            stage_i -= 1
            DDP_stage_last = stage_i
            DDP_long_stagetime = stageTime[DDP_stage_first]
            DDP_long_commDelay = commDelay[batch_action[DDP_stage_first]][batch_action[DDP_stage_first + 1]]
            for j in range(DDP_stage_last - DDP_stage_first):
                if stageTime[j + 1 + DDP_stage_first] > DDP_long_stagetime:
                    DDP_long_stagetime = stageTime[j + 1 + DDP_stage_first]
            for j in range(DDP_stage_last - DDP_stage_first - 1):
                if commDelay[batch_action[j + 1 + DDP_stage_first]][batch_action[j + 2 + DDP_stage_first]] > DDP_long_commDelay:
                    DDP_long_commDelay = stageTime[j + 1 + DDP_stage_first]
            for j in range(DDP_stage_last - DDP_stage_first + 1):
                stageTime[j + DDP_stage_first] = DDP_long_stagetime * 0.5 + DDP_long_commDelay * 2 * (DDP_stage_last - DDP_stage_first + 1 - 1) / (DDP_stage_last - DDP_stage_first + 1)
        stage_i += 1

    for i in range(len(stageComposition) - 1):
        if stageTime[i] < commDelay[batch_action[i]][batch_action[i + 1]]:
            stageTime[i] = commDelay[batch_action[i]][batch_action[i + 1]]

    maxStageTime = 0
    for i in range(len(stageComposition)):
        if stageComposition[i] > 0:
            epochTime += stageTime[i]
            if stageTime[i] > maxStageTime:
                maxStageTime = stageTime[i]
    for i in range(minibatch - 1):
        epochTime += maxStageTime

    return stageTime, epochTime


commDelay = np.array([
    [0, 0.1, 0.5, 0.5, 1, 1],
    [0.1, 0, 0.5, 0.5, 1, 1],
    [0.5, 0.5, 0, 0.1, 1.5, 1.5],
    [0.5, 0.5, 0.1, 0, 1.5, 1.5],
    [1, 1, 1.5, 1.5, 0, 0.1],
    [1, 1, 1.5, 1.5, 0.1, 0]
], dtype=np.float32)
