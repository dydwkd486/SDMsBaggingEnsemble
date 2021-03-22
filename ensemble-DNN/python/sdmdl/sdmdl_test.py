from sdmdl import sdmdl_main


import shutil
import os

for i in range(1,41):
    #  파일 위치 이동
    filename=str(i)+'.csv'
    print(filename)
    src = 'data/EasyEnsemble/'
    dir = 'data/spec_ppa_env/'
    shutil.move(src+filename, dir+'Hyla_japonica_total_env_dataframe_train.csv')
    #데이터 돌리기
    for j in range(1,2):
        model = sdmdl_main.sdmdl("F:/github/bird/code/sdmdl")
        # model.clean()
        # model.prep()
        model.train()
        # model.predict()
    # 결과 데이터 이름 변경
        src = 'data/results'
        dir = 'data/results'+str(j)
        os.rename(src,dir)
    # 폴더 위치 이동
        src = 'data/results' + str(j)
        dir = 'data/EasyEnsemble_results/data'+ str(i)+'/results' + str(j)
        shutil.move(src, dir)
    src = 'data/spec_ppa_env/Hyla_japonica_total_env_dataframe_train.csv'
    dir = 'data/EasyEnsemble_results/data'+ str(i)+'/Hyla_japonica_total_env_dataframe_train.csv'
    shutil.move(src, dir)


    # # 파일 삭제
    # filename = '5_Platalea_minor_result_env_dataframe_train.csv'
    # src = 'data/spec_ppa_env/'
    # os.remove(src + filename)

# model = sdmdl_main.sdmdl("F:/github/bird/code/sdmdl")
# model.clean()
# model.prep()
# model.train()
# model.predict()

'''
import pandas as pd

data = pd.read_csv("F:\github\sdmdl\data\gis\world_locations_to_predict1.csv",names=["decimal_longitude","decimal_latitude",""])
data =data.iloc[:,0:2]
data.to_csv("F:\github\sdmdl\data\gis\world_locations_to_predict.csv")
print(data.head())

'''