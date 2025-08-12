from __init__ import *
import utils as _U
reload(_U)


SUPPORTED_INDICATORS = ['MA']
#计算指标
def cal_indicators(tabular_df, indicator_name, parameters):
    if indicator_name == "MA":
        assert len(parameters) == 1, f'Wrong parameters num, expected 1, got {len(parameters)}'
        slice_win_size = int(parameters[0])
        MA = tabular_df['close'].rolling(slice_win_size, min_periods=1).mean()
        return MA # pd.Series



def single_symbol_image(tabular_df, image_size, start_date, sample_rate, indicators, show_volume, mode):
    ''' generate Candlelist images生成烛光图片
    
    parameters: [
        tabular_df  -> pandas.DataFrame: tabular data,原始数据集，dataframe形式
        image_size  -> tuple: (H, W), size shouble (32, 15), (64, 60)，输出的图像size
        start_date  -> int: truncate extra rows after generating images,从什么时间开始生成图像
        indicators  -> dict: technical indicators added on the image, e.g. {"MA": [20]},求指标
        show_volume -> boolean: show volume bars or not
        mode        -> 'train': for train & validation; 'test': for test; 'inference': for inference，用于训练，测试，推演的模式
    ]
    
    Note: A single day's data occupies 3 pixel (width). First rows's dates should be prior to the start date in order to make sure there are enough data to generate image for the start date.
    返回：
    return -> list: each item of the list is [np.array(image_size), binary, binary]. The last two binary (0./1.) are the label of ret5, ret20，列表的每一项都是[np.array(image_size)， binary, binary]。最后两个二进制数(0./1.)是ret5, ret20的标号
    
    '''
    
    
    ind_names = []
    if indicators:
        for i in range(len(indicators)//2):
            ind = indicators[i*2].NAME
            ind_names.append(ind)
            params = str(indicators[i*2+1].PARAM).split(' ')
            tabular_df[ind] = cal_indicators(tabular_df, ind, params)
    
    dataset = []
    valid_dates = []
    dateall=[]
    ##20天为一个窗口画图
    #图像部分除以3的整数部分，可能用作控制在处理图像时考虑的时间窗口或历史数据的数量，lookback=20=60//3
    lookback = image_size[1]//3
    
    ##for循环从d从19到len(tabular_df)-1
    for d in range(lookback-1, len(tabular_df)):
        '''
        这段代码的目的是在循环迭代处理数据时，以一定的概率跳过一些迭代，同时确保只处理在指定日期之后的数据
        '''
        # random skip some trading dates，随机生成一个随机数，跳过一些交易日期
        
        if np.random.rand(1) > sample_rate:
            continue
            
        # skip dates before start_date
        if tabular_df.iloc[d]['date'] < start_date:
            continue
        
        ##划分价格切片，时间间隔为d-(lookback-1):d+1，d=19,20,21,22,~
        ##[0:20]共0~19共20天为第一段绘制第一张图片，[1:21]为第二段，生成第二张图像，每20天划分一段,~[484-19:485]
        
        price_slice = tabular_df[d-(lookback-1):d+1][['open', 'high', 'low', 'close']+ind_names].reset_index(drop=True)
        volume_slice = tabular_df[d-(lookback-1):d+1][['volume']].reset_index(drop=True)
        ##新加
        date_first=tabular_df.iloc[d-(lookback-1)]['date']
        date_last=tabular_df.iloc[d]['date']
        
        '''
        这一部分检查在 price_slice 中，四个价格列的和是否等于开盘价的四倍，并统计符合条件的数量。
        如果数量超过 lookback（20）//5，则跳过当前迭代。这可能是用来过滤掉某些条件下的非交易日。
        是不是
        '''
        # number of no transactions days > 0.2*look back days
        if (1.0*(price_slice[['open', 'high', 'low', 'close']].sum(axis=1)/price_slice['open'] == 4)).sum() > lookback//5: 
            continue

        #####！！！！！有效的交易日期=最后一天日期，以筛选过后的一段窗口的第一天作为一张图片的日期
        valid_dates.append(tabular_df.iloc[d]['date']) 
        
        
        # trading dates surviving the validation
        
        # project price into quantile，将价格量和成交量归一化
        price_slice = (price_slice - np.min(price_slice.values))/(np.max(price_slice.values) - np.min(price_slice.values))
        volume_slice = (volume_slice - np.min(volume_slice.values))/(np.max(volume_slice.values) -np.min(volume_slice.values))

        if not show_volume:
            price_slice = price_slice.apply(lambda x: x*(image_size[0]-1)).astype(int)
        else:
            if image_size[0] == 32:
                price_slice = price_slice.apply(lambda x: x*(25-1)+7).astype(int)
                volume_slice = volume_slice.apply(lambda x: x*(6-1)).astype(int)
            else:
            ##图像是64×60的情况
            ##缩放了数据的范围：x*(51 - 1) + 13
                price_slice = price_slice.apply(lambda x: x*(51-1)+13).astype(int)
                volume_slice = volume_slice.apply(lambda x: x*(12-1)).astype(int)
        
        image = np.zeros(image_size)

        
        #画图以price_slice和volume_slice为基准
        ##一幅图像时间间隔的长度为len(price_slice)
        for i in range(len(price_slice)):
            # draw candlelist 
            image[price_slice.loc[i]['open'], i*3] = 255.
            image[price_slice.loc[i]['low']:price_slice.loc[i]['high']+1, i*3+1] = 255.
            image[price_slice.loc[i]['close'], i*3+2] = 255.
            # draw indicators
            for ind in ind_names:
                image[price_slice.loc[i][ind], i*3:i*3+2] = 255.
            # draw volume bars,成交量的图
            if show_volume:
                image[:volume_slice.loc[i]['volume'], i*3+1] = 255.

                
        ####上涨就是1，下跌就是0，正数返回1，负数返回0
        ##未来5/20天预计收益率大于0就是1
        label_ret5 = 1 if np.sign(tabular_df.iloc[d]['ret5']) > 0 else 0
        label_ret20 = 1 if np.sign(tabular_df.iloc[d]['ret20']) > 0 else 0
        label_code= tabular_df.iloc[d]['code']
        
        ##改！！
        entry = [image,label_ret5,label_ret20,date_first,date_last]
        dataset.append(entry)

        
    ##'train' 和'test'模式返回dataset格式：dataset=[image, label_ret5, label_ret20]
    if mode == 'train' or mode == 'test':
        return dataset
    ##其他：'inference'返回[tabular_df.iloc[0]['code'], dataset, valid_dates]格式
    elif mode == 'retrain':
        return [tabular_df.iloc[0]['code'], dataset, valid_dates]
    elif mode == 'inference':
        return [tabular_df.iloc[0]['code'], dataset, valid_dates]





        
############主要！！！
class ImageDataSet():
    def __init__(self, win_size, start_date, end_date, mode, label, indicators=[], show_volume=True, parallel_num=-1):
        ## Check whether inputs are valid
        assert isinstance(start_date, int) and isinstance(end_date, int), f'Type Error: start_date & end_date shoule be int'
        assert start_date < end_date, f'start date {start_date} cannnot be later than end date {end_date}'
        assert win_size in [5, 20], f'Wrong look back days: {win_size}'
        assert mode in ['train', 'test', 'inference','retrain'], f'Type Error: {mode}'
        assert label in ['RET5', 'RET20'], f'Wrong Label: {label}'
        assert indicators is None or len(indicators)%2 == 0, 'Config Error, length of indicators should be even'
        if indicators:
            for i in range(len(indicators)//2):
                assert indicators[2*i].NAME in SUPPORTED_INDICATORS, f"Error: Calculation of {indicators[2*i].NAME} is not defined"
        
        ## Attributes of ImageDataSet
        if win_size == 5:
            self.image_size = (32, 15)
            self.extra_dates = datetime.timedelta(days=40)
        else:
            self.image_size = (64, 60)
            self.extra_dates = datetime.timedelta(days=40)
            
        self.start_date = start_date
        self.end_date = end_date 
        self.mode = mode
        self.label = label
        self.indicators = indicators
        self.show_volume = show_volume
        self.parallel_num = parallel_num
        
        ## Load data from zipfile
        self.load_data()
        
        # Log info
        if indicators:
            ind_info = [(self.indicators[2*i].NAME, str(self.indicators[2*i+1].PARAM).split(' ')) for i in range(len(self.indicators)//2)]
        else:
            ind_info = []
        print(f"DataSet Initialized\n \t - Mode:         {self.mode.upper()}\n \t - Image Size:   {self.image_size}\n \t - Time Period:  {self.start_date} - {self.end_date}\n \t - Indicators:   {ind_info}\n \t - Volume Shown: {self.show_volume}\n \t - Label:   {self.label}")
        
        
    @_U.timer('Load Data', '8')
    ##下载数据集
    def load_data(self):
        if 'data' not in os.listdir():
            print('Download Original Tabular Data')
            os.system("mkdir data && cd data && wget 'https://cloud.tsinghua.edu.cn/f/f0bc022b5a084626855f/?dl=1' -O tabularDf.zip")
            
        if 'data' in os.listdir() and 'tabularDf.zip' not in os.listdir('data'):
            print('Download Original Tabular Data')
            os.system("cd data && wget 'https://cloud.tsinghua.edu.cn/f/f0bc022b5a084626855f/?dl=1' -O tabularDf.zip")
        ##改！！
        '''
        with ZipFile('data/tabularDf.zip', 'r') as z:
            f =  z.open('tabularDf.csv')
            ##新加
            f=z.open('/root/CNN-for-Trading/output_file1.csv')
            tabularDf = pd.read_csv(f, index_col=0)
            f.close()
            z.close()
        '''
        ##新加
        tabularDf = pd.read_csv('/root/CNN-for-Trading/output_file1.csv')
        # add extra rows to make sure image of start date and returns of end date can be calculated，计算开始日期和结束日期的返回值
        
        padding_start_date = int(str(pd.to_datetime(str(self.start_date)) - self.extra_dates).split(' ')[0].replace('-', ''))
        paddint_end_date = int(str(pd.to_datetime(str(self.end_date)) + self.extra_dates).split(' ')[0].replace('-', ''))
        #tabularDf是取开始和结束日期中间的数据
        self.df = tabularDf.loc[(tabularDf['date'] > padding_start_date) & (tabularDf['date'] < paddint_end_date)].copy(deep=False)
        tabularDf = [] # clear memory
        
        self.df['ret5'] = np.zeros(self.df.shape[0])
        self.df['ret20'] = np.zeros(self.df.shape[0])
        '''
        self.df['ret5']：整个表达式的含义是计算股票收盘价在过去5个交易日内的百分比变化，然后将这个变化百分比值向前移动5个位置，以便将每一天的行对应到未来5天的收益率。这种计算方式有助于分析股票的短期趋势。
        '''
        ##打标签
        #对于序列中的每个元素，pct_change计算方法为当前元素与前一个元素之间的百分比差异。
        self.df['ret5'] = (self.df['close'].pct_change(5)*100).shift(-5)
        self.df['ret20'] = (self.df['close'].pct_change(20)*100).shift(-20)
        ##保留DataFrame中日期列小于或等于self.end_date的所有行，删除不符合条件的行，从而获得一个在指定日期（end_date）之前的子集。
        self.df = self.df.loc[self.df['date'] <= self.end_date]
        
    ###调用了single_symbol_image
    def generate_images(self, sample_rate):
        dataset_all = Parallel(n_jobs=self.parallel_num)(delayed(single_symbol_image)(\
                                        g[1], image_size = self.image_size,\
                                           start_date = self.start_date,\
                                          sample_rate = sample_rate,\
                                           indicators = self.indicators,\
                                          show_volume = self.show_volume, \
                                            mode = self.mode
                                        ) for g in tqdm(self.df.groupby('code'), desc=f'Generating Images (sample rate: {sample_rate})'))
        
        #按照 'code' 列的不同取值进行分组，然后在每个分组上执行一些操作（生成图像），同时使用 tqdm 显示循环的进度条，让用户能够看到代码的执行进度。
        #if self.mode == 'train' or self.mode == 'test' or self.mode == 'retrain':
        if self.mode == 'train' or self.mode == 'test':
            image_set = []
            for symbol_data in dataset_all:
                image_set = image_set + symbol_data
            print("good:")
            dataset_all = [] # clear memory
            
            #if self.mode == 'train' or self.mode == 'retrain': # resample to handle imbalance
            if self.mode == 'train':
                ##改！！
                print("good")
                image_set = pd.DataFrame(image_set, columns=['img', 'ret5', 'ret20','date_first','date_last','label_code'])
                image_set['index'] =  image_set.index
                smote = SMOTE()
                '''
                确保了生成的合成样本的顺序与原始样本相同。这样的做法通常是为了保持样本的时间顺序或其他特定顺序的一致性。
                '''
                if self.label == 'RET5':
                    num0_before = image_set.loc[image_set['ret5'] == 0].shape[0]
                    num1_before = image_set.loc[image_set['ret5'] == 1].shape[0]
                    ##这一行使用 SMOTE 过采样算法对选定的特征列（'index' 和 'ret20'）和目标列（'ret5'）进行过采样。生成的合成样本会按照原始样本的顺序排列，以保持与原始数据的对应关系。
                    resample_index, _ = smote.fit_resample(image_set[['index', 'ret20']], image_set['ret5'])
                    #改！！
                    #image_set = image_set[['img', 'ret5', 'ret20']].loc[resample_index['index']]
                    image_set = image_set[['img', 'ret5', 'ret20','date_first','date_last','label_code']].loc[resample_index['index']]
                    num0 = image_set.loc[image_set['ret5'] == 0].shape[0]
                    num1 = image_set.loc[image_set['ret5'] == 1].shape[0]
                    image_set = image_set.values.tolist()
                    
                else:
                    num0_before = image_set.loc[image_set['ret20'] == 0].shape[0]
                    num1_before = image_set.loc[image_set['ret20'] == 1].shape[0]
                    resample_index, _ = smote.fit_resample(image_set[['index', 'ret5']], image_set['ret20'])
                    image_set = image_set[['img', 'ret5', 'ret20','date_first','date_last','label_code']].loc[resample_index['index']]
                    num0 = image_set.loc[image_set['ret20'] == 0].shape[0]
                    num1 = image_set.loc[image_set['ret20'] == 1].shape[0]
                    image_set = image_set.values.tolist()
                print("good2")
                print(f"LABEL: {self.label}\n\tBefore Resample: 0: {num0_before}/{num0_before+num1_before}, 1: {num1_before}/{num0_before+num1_before}\n\tResampled ImageSet: 0: {num0}/{num0+num1}, 1: {num1}/{num0+num1}")
                 
            
            return image_set
    
        else:
            return dataset_all