import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import warnings
warnings.filterwarnings('ignore')

# df_mau = pd.read_csv("/home/tran.xuan.tien@sun-asterisk.com/Sensory/NIR/gop_file.csv", encoding='utf-8-sig')
# columns = df_mau.columns

# folder_path = "/home/tran.xuan.tien@sun-asterisk.com/Sensory/NIR/Dak_lak"
# dir_path = os.path.dirname(os.path.realpath(folder_path))
# folder_names = np.array([f for f in os.listdir(folder_path)])

# for folder_name in folder_names:
#     folder_names_1 = np.array(os.listdir(folder_path+'/'+folder_name))
    
#     for folder_name_1 in folder_names_1:
#         folder_names_2 = np.array(os.listdir(folder_path+'/'+folder_name+'/'+folder_name_1))

#         for folder_name_2 in folder_names_2:
#             file_names = np.array(os.listdir(folder_path+'/'+folder_name+'/'+folder_name_1+'/'+folder_name_2))

#             for file_name in file_names:
#                 endfile = str(file_name[-3:])
#                 value = []
#                 if(endfile == 'csv'):
#                     x = np.asarray(folder_name_1).reshape(1, -1)
#                     y = np.asarray(folder_name_2).reshape(1, -1)
#                     #z = np.asarray(file_name[7:8]).reshape(1, -1)
#                     file = pd.read_csv(folder_path+'/'+folder_name+'/'+folder_name_1+'/'+folder_name_2+'/'+file_name, error_bad_lines=False)
#                     time = file.iloc[0, 1]
#                     a = time.replace(' @ ', ' ')
#                     a = pd.to_datetime(a)
#                     a = np.asarray(a).reshape(1, -1)
#                     test = pd.read_csv(folder_path+'/'+folder_name+'/'+folder_name_1+'/'+folder_name_2+'/'+file_name,skiprows = 21)
#                     test = np.array(test['Absorbance (AU)']).reshape(1, -1)
#                     value = np.concatenate((x,y,test), axis = 1)
#                     value = pd.DataFrame(value, columns=columns)
#                     df_mau = df_mau.append(value, ignore_index=True)
#                     df_mau.to_csv (r"/home/tran.xuan.tien@sun-asterisk.com/Sensory/NIR/gop_file.csv", index = None, header=True)    
# df_mau.to_csv("/home/tran.xuan.tien@sun-asterisk.com/Sensory/NIR/gop_file.csv", index = False, encoding='utf-8-sig')

# df= pd.read_csv("/home/tran.xuan.tien@sun-asterisk.com/Sensory/NIR/data.csv", encoding='utf-8-sig')

# df_histamin = pd.read_csv("/home/tran.xuan.tien@sun-asterisk.com/Sensory/NIR/gop_file.csv", encoding='utf-8-sig')
# #print(df,df_histamin)

# #df.merge(df_histamin, on='ma mau').drop(['STT'],axis=1)
# #merged_df = df1.merge(df2, on=['key1', 'key2'], how='inner')

# ctv3 = pd.merge(df, df_histamin, on='ma_mau')
# print(ctv3)

# ctv3.to_csv("/home/tran.xuan.tien@sun-asterisk.com/Sensory/NIR/full.csv", index = False, encoding='utf-8-sig')




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
import os, re
import seaborn as sns
from sys import stdout

from scipy.signal import savgol_filter
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("/home/tran.xuan.tien@sun-asterisk.com/Sensory/NIR/full.csv")
data = data.replace([np.inf, -np.inf], np.nan).dropna()  # loại NaN/Inf

y = data.values[:, -5].astype(float)
X = data.values[:, 3:-5].astype(float)

x2 = savgol_filter(X, 15, polyorder=2, deriv=2) 
def pls_variable_selection(x2, y, max_comp, n_cv):
    
# Khai bao mảng de luu gia tri MSE khi loại bỏ dần dữ liệu 
    mse = np.zeros((max_comp,x2.shape[1]))
        # Lap gia tri 
    for i in range(max_comp):
            
            # Chạy hồi quy PLS lần 1 với component = i + 1 vì i bắt đầu từ 0
        pls1 = PLSRegression(n_components=i+1)
        pls1.fit(x2, y)

            # Lấy thứ tự các bước sóng theo thứ tự từ thấp tới cao theo trị tuyệt đối của PLS coefficients
        sorted_wlt = np.argsort(np.abs(pls1.coef_.flatten()))
    
            # Sắp xếp lại quang phổ theo thứ tự các bước sóng quan trọng đã lấy ở trên
        Xc = x2[:,sorted_wlt]
    
            # Discard one wavelength at a time of the sorted spectra,
            # regress, and calculate the MSE cross-validation

            # Loại bỏ dần từng bước sóng theo thứ tự đã sắp xếp
        for j in range(150, Xc.shape[1]-(i+1)):

                # Hồi quy lần 2 với input là ma trận các bước sóng còn lại và y 
            pls2 = PLSRegression(n_components=i+1)
            pls2.fit(Xc[:, j:], y)
                # Tính giá trị xác thực chéo 
            y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=n_cv)
    
            mse[i,j] = mean_squared_error(y, y_cv)
    
        # Tìm giá trị MSE min trong mảng 2 chiều (chiều x là số component tối ưu, chiều y là số bước sóng loại bỏ )
    mseminx, mseminy = np.where(mse==np.min(mse[np.nonzero(mse)]))
    
    print("PLS components: ", mseminx[0]+1)
    print("Number Wavelengths remove: ",mseminy[0])
    print('MSE min: ', mse[mseminx,mseminy][0])
    stdout.write("\n")
    
        # Calculate PLS with optimal components and export values
        # Tính lại kết quả để return
    pls = PLSRegression(n_components=mseminx[0]+1)
    pls.fit(x2, y)
        # Thứ tự bước sóng được sắp xếp
    sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))
        # Ma trận bươc sóng đã được loại bỏ
    Xc = x2[:,sorted_ind][:,mseminy[0]:]
        
        
    # def optimise_pls_cv(Xc, y, n_comp):
    #     # Define PLS object
    #     pls = PLSRegression(n_components=n_comp)
    #     pls.fit(Xc, y)
    #     y_c = pls.predict(Xc)
    #         # Cross-validation
    #     y_cv = cross_val_predict(pls, Xc, y, cv=10)

    #         # Calculate scores
    #     r2_cv = r2_score(y, y_cv)
    #     r2_calib = r2_score(y, y_c)
    #     mse_cv = mean_squared_error(y, y_cv)
    #     mse_calib = mean_squared_error(y, y_c)
    #     rpd = y.std()/np.sqrt(mse)
            
    #     return (y_cv, r2_cv, mse, rpd)

    # y_cv, r2_cv, mse, rpd = optimise_pls_cv(Xc, y, mseminx[0]+1) #n-compt
    # print('R2_CV: %0.4f, R2_Calib: %0.4f , MSE_CV: %0.4f, MSE_Calib: %0.4f , RPD: %0.4f' %(r2_cv, r2_calib, mse_cv, mse_calib, rpd))
    
    pls_opt = PLSRegression(n_components=mseminx[0]+1)
 
    # Fir to the entire dataset
    pls_opt.fit(Xc, y)
    y_c = pls_opt.predict(Xc)
 
    # Cross-validation
    y_cv = cross_val_predict(pls_opt, Xc, y, cv=10)
 
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
 
    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
 
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
         
    y = y.reshape(-1).astype(float)
    y_cv = y_cv.reshape(-1).astype(float)

        


    plt.figure(figsize=(6, 6))
    with plt.style.context('ggplot'):
        plt.scatter(y, y_cv, color='red')
        plt.plot(y, y, '-g', label='Expected regression line')
        z = np.polyfit(y, y_cv, 1)
        plt.plot(np.polyval(z, y), y, color='blue', label='Predicted regression line')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.plot()
        plt.show()

    # Xsong1 =[]
    # for k in range(mseminy[0], 228):
    #     Xsong = sorted_ind[k]
    #     #ctv3 = pd.join(Xsong)
    #     Xsong1.append(Xsong)
    #     df1 = pd.DataFrame(Xsong1)
    #     df1.to_csv("D:/NIR/phan-tich_buoc_song/buoc_song_120_228/test.csv", index = False, encoding='utf-8-sig') 
    #     print(Xsong1)
    
    return(Xc,mseminx[0]+1,mseminy[0], sorted_ind)

plt.show()
pls_variable_selection(x2, y, 40, 10)