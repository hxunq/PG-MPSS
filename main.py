from FirstNet import FirstNet
import SetReward
from SecondNet import SecondNet
from SetReward import getStageTime
import show_result
from InputInfo.InceptionV3Info import InceptionV3Info as ModelInfo

layer_num = len(ModelInfo().getModelInfo())


def train(FN, SN, epoch_n):
    print("begin training FN & SN ......")
    device_n = len(ModelInfo().getExeTime())
    for epoch in range(epoch_n):
        FNoutput, FNbatch_action = FN.doActions()
        if FN.countAction(batch_action=FNbatch_action, batch_n=layer_num) > device_n:
            FNreward_tensor = SetReward.get_reward_tensor3(batch_n=layer_num, batch_action=FNbatch_action)
            FN.update(model_output=FNoutput, batch_action=FNbatch_action, reward_tensor=FNreward_tensor)
            continue
        else:
            pre_handle_DNNInfo, stageComposition = SN.inputInfo_pre_handle(FNbatch_action=FNbatch_action, layer_num=layer_num, device_n=device_n)
            SNoutput, SNbatch_action = SN.doOtherActions(pre_handle_DNNInfo)

        stageTime, epochTime = getStageTime(batch_action=SNbatch_action, stageComposition=stageComposition)
        FNreward_tensor = SetReward.get_reward_tensor4(batch_n=layer_num, stageComposition=stageComposition, stageTime=stageTime, epochTime=epochTime)
        SNreward_tensor = SetReward.get_reward_tensor5(stageTime=stageTime, epochTime=epochTime)
        FN.update(model_output=FNoutput, batch_action=FNbatch_action, reward_tensor=FNreward_tensor)
        SN.update(model_output=SNoutput, batch_action=SNbatch_action, reward_tensor=SNreward_tensor)

        if epoch % 500 == 0:
            FN_action = FN.doGreedyActions()
            if FN.countAction(batch_action=FN_action, batch_n=layer_num) > device_n:
                print("epoch:", epoch, " expl_epochTime:{:.2f}".format(epochTime))
                continue
            pre_handle_DNNInfo, stageComposition = SN.inputInfo_pre_handle(FNbatch_action=FN_action, layer_num=layer_num, device_n=device_n)
            SNbatch_action = SN.doOtherGreedyActions(pre_handle_DNNInfo)
            stageTime, use_epochTime = getStageTime(batch_action=SNbatch_action, stageComposition=stageComposition)
            print("epoch:", epoch, " expl_epochTime:{:.2f}".format(epochTime), " use_epochTime:{:.2f}".format(use_epochTime))
    return FN, SN


def trainSN(FN, SN, epoch_n):
    print("begin training SN ......")
    FN_action = FN.doGreedyActions()
    device_n = len(ModelInfo().getExeTime())
    sum_reward_SN = 0

    for epoch in range(epoch_n):
        pre_handle_DNNInfo, stageComposition = SN.inputInfo_pre_handle(FNbatch_action=FN_action, layer_num=layer_num, device_n=device_n)
        SNoutput, SNbatch_action = SN.doOtherActions(pre_handle_DNNInfo)
        stageTime, epochTime = getStageTime(batch_action=SNbatch_action, stageComposition=stageComposition)
        SNreward_tensor, sum_reward_SN = SetReward.get_reward_tensor7(stageTime=stageTime, epochTime=epochTime, sum_reward_SN=sum_reward_SN, epoch=epoch)
        SN.update(model_output=SNoutput, batch_action=SNbatch_action, reward_tensor=SNreward_tensor)

        if epoch % 500 == 0:
            SNbatch_action = SN.doOtherGreedyActions(pre_handle_DNNInfo)
            stageTime, use_epochTime = getStageTime(batch_action=SNbatch_action, stageComposition=stageComposition)
            print("epoch:", epoch + 150000, " expl_epochTime:{:.2f}".format(epochTime), " use_epochTime:{:.2f}".format(use_epochTime))

    return FN, SN


def load(FN, SN):
    FN.loadFN_Parameters()
    SN.loadSN_Parameters()
    return FN, SN


def save(FN, SN):
    FN.saveFN_Parameters()
    FN.saveFN_All()
    SN.saveSN_Parameters()
    SN.saveSN_All()
    print("save... success")


def domain():
    inputInfo = ModelInfo().getModelInfo()
    device_n = len(ModelInfo().getExeTime())

    FN = FirstNet(batch_n=layer_num, input_data_size=40, output_data_size=2, lr=5e-6, inputInfo=inputInfo, cuda="cuda:0", device_n=device_n)
    SN = SecondNet(input_data_size=40, output_data_size=device_n, lr=7e-6, DNNInfo=inputInfo, batch_n=device_n, cuda="cuda:0")

    # FN, SN = load(FN, SN)

    FN, SN = train(FN, SN, epoch_n=150000)
    show_result.predict1(FN, SN)

    SN = SecondNet(input_data_size=40, output_data_size=device_n, lr=8e-6, DNNInfo=inputInfo, batch_n=device_n, cuda="cuda:0")
    FN, SN = trainSN(FN, SN, epoch_n=150000)
    show_result.predict1(FN, SN)

    # save(FN, SN)


if __name__ == '__main__':
    domain()
