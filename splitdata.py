###################
from torch.utils.data.dataset import Subset
def split_train_valid(retrain_image_set):
    #ka_all存储每种股票的训练集图片数
    ka_all=[]
    #ka_all存储每种股票的所有图片数
    num_all=[]
    #len(retrain_image_set)=期货的种数
    for j in range(len(retrain_image_set)):
        #每种股票的图片数
        num=len(retrain_image_set[j][2])
        num_all.append(num)
        ##如果num==0代表这种股票不在指定区间内，跳过
        if num==0:
            if j==0:
                b=1
            ka=0
            ka_all.append(ka)
            continue
        else:
            ka=0
            for i in range (num):
                #开始日期
                a=retrain_image_set[j][1][i][3]
                key_date=20220701
                if a<key_date:
                    ka=ka+1
            #ratio=ka/num
            ka_all.append(ka)
            #对每种股票划分训练测试集
            key=retrain_image_set[j][1]
            #key_train_loader_size = int(len(key)*(ratio))
            key_train_loader_size=ka
            #key_valid_loader_size = len(key) - key_train_loader_size
            key_valid_loader_size=num-key_train_loader_size
            #划分训练测试集
            key_train_loader = Subset(retrain_image_set[j][1], range(key_train_loader_size))
            key_valid_loader = Subset(retrain_image_set[j][1], range(key_train_loader_size, key_train_loader_size + key_valid_loader_size))
            #key_train_loader, key_valid_loader = torch.utils.data.random_split(key, [key_train_loader_size, key_valid_loader_size])
            #j==0时，用于初始化
            if j==0:
                key_all_train_loader=key_train_loader
                key_all_valid_loader=key_valid_loader
            else:
            #累加
                if b==1:
                    key_all_train_loader=key_train_loader
                    key_all_valid_loader=key_valid_loader 
                    b=0
                else:
                    key_all_train_loader=key_all_train_loader+key_train_loader
                    key_all_valid_loader=key_all_valid_loader+key_valid_loader   
    return key_all_train_loader,key_all_valid_loader,key_date

if __name__ == '__main__':
    split_train_valid(retrain_image_set)