import random
from SetReward import getStageTime
from InputInfo.InceptionV3Info import InceptionV3Info as ModelInfo

layer_num = len(ModelInfo().getModelInfo())


def predict1(FN, SN):
    FNbatch_action = FN.doGreedyActions()
    device_n = len(ModelInfo().getExeTime())
    pre_handle_DNNInfo, stageComposition = SN.inputInfo_pre_handle(FNbatch_action, layer_num, device_n)
    SNbatch_action = SN.doOtherGreedyActions(pre_handle_DNNInfo)

    stageTime, epochTime = getStageTime(batch_action=SNbatch_action, stageComposition=stageComposition)
    print("************************************")
    print("predict：")
    print("FNbatch_action：", FNbatch_action)
    print("stageComposition:", stageComposition)
    print("SNbatch_action:", SNbatch_action)
    print("epochTime:", epochTime)
    print("************************************")


def predict2(FN, SN):
    sum_epochTime = 0
    best_epochTime = 10000
    device_n = len(ModelInfo().getExeTime())
    for i in range(100000):
        FNoutput, FNbatch_action = FN.doActions()
        if FN.countAction(batch_action=FNbatch_action, batch_n=layer_num) > device_n:
            continue
        pre_handle_DNNInfo, stageComposition = SN.inputInfo_pre_handle(FNbatch_action, layer_num, device_n)
        SNoutput, SNbatch_action = SN.doOtherActions(pre_handle_DNNInfo)
        stageTime, epochTime = getStageTime(batch_action=SNbatch_action, stageComposition=stageComposition)

        sum_epochTime += epochTime
        if best_epochTime > epochTime:
            best_epochTime = epochTime
    print("ave_epochTime = ", sum_epochTime / 100000)
    print("best_epochTime = ", best_epochTime)


def predict3(FN, SN, dev):
    device_n = len(ModelInfo().getMutilExeTime()[dev])
    FNbatch_action = FN.doGreedyActions()
    if FN.countAction(batch_action=FNbatch_action, batch_n=layer_num) <= device_n:
        pre_handle_DNNInfo, stageComposition = SN.inputInfo_pre_handle(FNbatch_action, layer_num, device_n)
        SNbatch_action = SN.doOtherGreedyActions(pre_handle_DNNInfo)

        stageTime, epochTime = getStageTime(batch_action=SNbatch_action, stageComposition=stageComposition)

        print("epochTime:", epochTime)
