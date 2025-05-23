import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import pandas as pd

index = ['Token', 'Para',  'Misspelling', 'Typo',  'Lowercase',  'Contraction',  'Expansion', 'Modify',  'Synonym',  'Translate']
index2 = ['Token', 'Para',  'Misspelling', 'Typo',  'Lowercase',  'Contraction',  'Expansion', 'Modify',  'Synonym',  'Translate']
data=pd.DataFrame(pd.read_csv('/data1/lzs/MarkText/watermark/UniSpach/multi_detect.csv',index_col=0, header=0, names=index))
data.reset_index(drop=True, inplace=True)
data.index = pd.Index(index2)
data2 = pd.DataFrame(pd.read_csv('/data1/lzs/MarkText/watermark/KGW/multi_detect.csv',index_col=0,header=0, names=index))
data2.reset_index(drop=True, inplace=True)
data2.index = pd.Index(index2)
# data_str = """
# '' Token Para  Misspelling Typo  Lowercase  Contraction  Expansion Modify  Synonym  Translate 
# Token 0.0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
# Para 0.0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
# Misspelling 0.0 0.003378 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
# Typo 0.0 0.006757 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
# Lowercase 0.0 0.010135 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.003378 0.000000 
# Contraction 0.0 0.006757 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.003378 0.000000 
# Expansion 0.0 0.013514 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.003378 0.000000 
# Modify 0.0 0.013514 0.000000 0.0000   00 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
# Synonym 0.0 0.016892 0.000000 0.003378 0.003378 0.003378 0.003378 0.003378 0.010135 0.000000 
# Translate 0.0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
# """

# data2_str = """
# '' Token Para  Misspelling Typo Lowercase  Contraction  Expansion Modify  Synonym  Translate 
# Token 0.520270 0.000000 0.520270 0.520270 0.520270 0.523649 0.516892 0.287162 0.287162 0.000000
# Para 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
# Misspelling 0.750000 0.000000 0.750000 0.750000 0.750000 0.750000 0.746622 0.516892 0.516892 0.000000
# Typo 0.652027 0.000000 0.652027 0.652027 0.652027 0.658784 0.648649 0.456081 0.456081 0.000000
# Lowercase 0.682432 0.000000 0.682432 0.682432 0.682432 0.689189 0.682432 0.483108 0.479730 0.000000
# Contraction 0.679054 0.000000 0.679054 0.679054 0.679054 0.679054 0.672297 0.456081 0.456081 0.000000
# Expansion 0.722973 0.000000 0.722973 0.722973 0.722973 0.726351 0.722973 0.506757 0.506757 0.000000
# Modify 0.506757 0.000000 0.516892 0.516892 0.516892 0.520270 0.503378 0.253378 0.256757 0.000000
# Synonym 0.425676 0.000000 0.425676 0.425676 0.425676 0.429054 0.425676 0.236486 0.236486 0.000000
# Translate 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
# """
# data_str = """
# ''	Token	Para	Misspelling	Typo	Lowercase	Contraction	Expansion	Modify	Synonym	Translate
# Token	0.315 	0.651 	0.305 	0.255 	0.294 	0.294 	0.298 	0.293 	0.291 	0.642 
# Para	0.655 	0.606 	0.598 	 0.553	0.567 	0.549 	0.544 	0.613 	0.549 	0.624 
# Misspelling	0.309 	0.629 	0.316 	0.264 	0.297 	0.295 	0.298 	0.288 	0.286 	0.640 
# Typo	0.233 	0.613 	0.229 	0.183 	0.251 	0.249 	0.248 	0.239 	0.239 	0.499 
# Lowercase	0.285 	0.596 	0.280 	0.249 	0.247 	0.253 	0.252 	0.251 	0.248 	0.551 
# Contraction	0.279 	0.593 	0.282 	0.256 	0.255 	0.256 	0.252 	0.257 	0.256 	0.554 
# Expansion	0.279 	0.599 	0.279 	0.249 	0.246 	0.254 	0.253 	0.244 	0.244 	0.554 
# Modify	0.317 	0.645 	0.302 	0.256 	0.295 	0.295 	0.296 	0.295 	0.284 	0.670 
# Synonym	0.280 	0.503 	0.275 	0.242 	0.246 	0.245 	0.242 	0.236 	0.239 	0.507 
# Translate	0.621 	0.643 	0.520 	0.505 	0.516 	0.531 	0.424 	0.548 	0.515 	0.667 
# """
# data2_str = """
# ''	Token	Para	Misspelling	Typo	Lowercase	Contraction	Expansion	Modify	Synonym	Translate
# Token	0.348141892	0.674662162	0.320945946	0.285810811	0.302702703	0.303378378	0.303547297	0.303716216	0.298986486	0.700337838
# Para	0.646283784	0.656756757	0.613851351	0.568243243	0.567905405	0.564864865	0.569932432	0.572635135	0.568918919	0.658445946
# Misspelling	0.323141892	0.679054054	0.321114865	0.286993243	0.297804054	0.296621622	0.300168919	0.29375	0.294594595	0.681081081
# Typo	0.265878378	0.632432432	0.264527027	0.248310811	0.276351351	0.272804054	0.273648649	0.277027027	0.276689189	0.507432432
# Lowercase	0.289864865	0.590878378	0.289189189	0.275168919	0.274662162	0.273648649	0.277195946	0.275337838	0.271452703	0.555405405
# Contraction	0.291047297	0.603716216	0.290709459	0.275844595	0.274493243	0.275	0.278378378	0.280405405	0.270439189	0.550337838
# Expansion	0.294594595	0.599324324	0.291722973	0.280236486	0.278716216	0.276689189	0.277533784	0.276858108	0.273817568	0.558108108
# Modify	0.289695946	0.566216216	0.289358108	0.281587838	0.278885135	0.281587838	0.282094595	0.277027027	0.275168919	0.547297297
# Synonym	0.275675676	0.462837838	0.277871622	0.263851351	0.261655405	0.26402027	0.264527027	0.258783784	0.260472973	0.523648649
# Translate	0.689864865	0.6625	0.633783784	0.587837838	0.601013514	0.602364865	0.603716216	0.597635135	0.587162162	0.745945946
# """
# 使用StringIO将字符串转换为文件对象
from io import StringIO
# data = pd.read_csv(StringIO(data_str),index_col=0)
# data2 = pd.read_csv(StringIO(data2_str),index_col=0)
sns.set(style="white")
# mask = np.ones((11,9))
# for i in range(1,11):
#     for j in range(1,i + 1):
#         if j < 9:
#             mask[i,j] = 0
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
sns.heatmap(data, annot=True, fmt=".4f", linewidths=1.5, ax=axes[0], cmap='YlGnBu',square=True,cbar = False,annot_kws= {"fontsize": 16})
sns.heatmap(data2, annot=True, fmt=".4f", linewidths=1.5, ax=axes[1], cmap="YlGnBu", square=True, cbar=False,annot_kws= {"fontsize": 16})
axes[0].set_title('KGW')
axes[1].set_title('UniSpach')
plt.tight_layout()
plt.subplots_adjust(wspace=0.02)  
# sns.heatmap(data=data,cmap='YlGnBu',annot=True,linewidth=0.9,linecolor='white',
#             cbar=True,vmax=None,vmin=None,center=0,square=True,
#             # mask=mask,robust=True,
#             annot_kws={'color':'white','size':1,'family':None,'style':None,'weight':10},
#             cbar_kws={'orientation':'vertical','shrink':1,'extend':'max','location':'right'})
plt.show()
plt.savefig("/home/lsz/MarkText/image/1.png", dpi = 500)