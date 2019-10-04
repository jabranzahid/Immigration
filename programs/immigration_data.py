"""
This module is meant to read in the subsetted immigration data.
User must set path = 'Users/jabran/insight/project/data/subset/'
to appropriate path on their machine. All data was extracted from
GIS sources using a vector overlay. 
Needs pandas version 0.25.1
Warning: Do not set nyears_lookback to anything other than 1 or 3 unless
you modify the code appropriately. 
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


pd.options.mode.chained_assignment = None

#This is dangerous to do and bad practice,
#but doing anyways!!!!s
import warnings
warnings.filterwarnings("ignore")



class immigration_data:

    def __init__(self, standardize = True, path = '/Users/jabran/insight/project/data/subset/',
                 nyears_lookback = 1, year_min = 2001, classify = True,
                ):

        self.standardize = standardize
        self.path = path
        self.get_id()
        self.read_admin_boundaries()
        self.nyears_lookback = nyears_lookback
        self.year_min = year_min
        self.classify = classify

        
    def get_id(self):

        file_ = 'population_subset.csv'
        file = self.path + file_
        pop = pd.read_csv(file)
        self.id = pop['id']


    def read_admin_boundaries(self):

        file_ = 'admin_boundaries_subset.csv'
        file = self.path + file_
        admin = pd.read_csv(file)

        id_ind = (np.unique(admin['id'], return_index = True))
        country = admin['NAME_0'][id_ind[1]]
        self.country = np.asarray(country)


        
    def remove_pop_growth_trend(self, pop):
        
        #Remove average population growth 
        
        col_name = []
        year = np.arange(16)+2002
        for yy in year:
            col_name.append(str(yy) + '_mean')        
        
        avg_pop = pop[col_name].sum()/pop[col_name[-1]].sum()
        ll = np.polyfit(year, avg_pop, 1)
        
        pop['2000_mean'] /= (ll[0]*2000 + ll[1])
        pop['2001_mean'] /= (ll[0]*2001 + ll[1])
        pop['2002_mean'] /= (ll[0]*2002 + ll[1])
        pop['2003_mean'] /= (ll[0]*2003 + ll[1])
        pop['2004_mean'] /= (ll[0]*2004 + ll[1])
        pop['2005_mean'] /= (ll[0]*2005 + ll[1])
        pop['2006_mean'] /= (ll[0]*2006 + ll[1])
        pop['2007_mean'] /= (ll[0]*2007 + ll[1])
        pop['2008_mean'] /= (ll[0]*2008 + ll[1])
        pop['2009_mean'] /= (ll[0]*2009 + ll[1])
        pop['2010_mean'] /= (ll[0]*2010 + ll[1])
        pop['2011_mean'] /= (ll[0]*2011 + ll[1])
        pop['2012_mean'] /= (ll[0]*2012 + ll[1])
        pop['2013_mean'] /= (ll[0]*2013 + ll[1])
        pop['2014_mean'] /= (ll[0]*2014 + ll[1])
        pop['2015_mean'] /= (ll[0]*2015 + ll[1])
        pop['2016_mean'] /= (ll[0]*2016 + ll[1])
        pop['2017_mean'] /= (ll[0]*2017 + ll[1])
        
            
    def read_population(self, remove_trend = True):

        #generate both difference and ratio of population for two consecutive years
        
        file_ = 'population_subset.csv'
        file = self.path + file_
        pop = pd.read_csv(file)

        pop['2001_mean'] = (pop['2000_mean'] + pop['2002_mean'])/2

        if remove_trend: self.remove_pop_growth_trend(pop)
        
        
        pop['dp_2017'] = pop['2017_mean'] / pop['2016_mean']
        pop['dp_2016'] = pop['2016_mean'] / pop['2015_mean']
        pop['dp_2015'] = pop['2015_mean'] / pop['2014_mean']
        pop['dp_2014'] = pop['2014_mean'] / pop['2013_mean']
        pop['dp_2013'] = pop['2013_mean'] / pop['2012_mean']
        pop['dp_2012'] = pop['2012_mean'] / pop['2011_mean']
        pop['dp_2011'] = pop['2011_mean'] / pop['2010_mean']
        pop['dp_2010'] = pop['2010_mean'] / pop['2009_mean']
        pop['dp_2009'] = pop['2009_mean'] / pop['2008_mean']
        pop['dp_2008'] = pop['2008_mean'] / pop['2007_mean']
        pop['dp_2007'] = pop['2007_mean'] / pop['2006_mean']
        pop['dp_2006'] = pop['2006_mean'] / pop['2005_mean']
        pop['dp_2005'] = pop['2005_mean'] / pop['2004_mean']
        pop['dp_2004'] = pop['2004_mean'] / pop['2003_mean']
        pop['dp_2003'] = pop['2003_mean'] / pop['2002_mean']
        pop['dp_2002'] = pop['2002_mean'] / pop['2001_mean']
        pop['dp_2001'] = pop['2001_mean'] / pop['2000_mean']

        pop['dd_2017'] = pop['2017_mean'] - pop['2016_mean']
        pop['dd_2016'] = pop['2016_mean'] - pop['2015_mean']
        pop['dd_2015'] = pop['2015_mean'] - pop['2014_mean']
        pop['dd_2014'] = pop['2014_mean'] - pop['2013_mean']
        pop['dd_2013'] = pop['2013_mean'] - pop['2012_mean']
        pop['dd_2012'] = pop['2012_mean'] - pop['2011_mean']
        pop['dd_2011'] = pop['2011_mean'] - pop['2010_mean']
        pop['dd_2010'] = pop['2010_mean'] - pop['2009_mean']
        pop['dd_2009'] = pop['2009_mean'] - pop['2008_mean']
        pop['dd_2008'] = pop['2008_mean'] - pop['2007_mean']
        pop['dd_2007'] = pop['2007_mean'] - pop['2006_mean']
        pop['dd_2006'] = pop['2006_mean'] - pop['2005_mean']
        pop['dd_2005'] = pop['2005_mean'] - pop['2004_mean']
        pop['dd_2004'] = pop['2004_mean'] - pop['2003_mean']
        pop['dd_2003'] = pop['2003_mean'] - pop['2002_mean']
        pop['dd_2002'] = pop['2002_mean'] - pop['2001_mean']
        pop['dd_2001'] = pop['2001_mean'] - pop['2000_mean']
        
        return pop


    def read_arms_exports(self):

        file_ = 'Arms_Export_ID_0.csv'
        file = self.path + file_
        arms = pd.read_csv(file)

        arms_guatemala = arms.loc[(arms['Country'] == 'Guatemala') & (arms['Recipient'] == 'Guatemala')]
        arms_honduras = arms.loc[(arms['Country'] == 'Honduras') ]
        arms_elsalvador = arms.loc[(arms['Country'] == 'El Salvador') & (arms['Recipient'] == 'El Salvador') ]

        arms_out = pd.concat([arms_guatemala, arms_honduras, arms_elsalvador], keys=arms.keys())

        df_out = pd.DataFrame({'id':self.id, 'Country' : self.country})
        arms_merge = pd.merge(df_out, arms_out, on='Country')
        arms_merge = arms_merge.sort_values('id')

        arms_merge = arms_merge.fillna(0)

        return arms_merge

    def read_rain(self):

        #.ti is count, '_1' is sum , '_2' is mean which is what we want

        file_ = 'CHIRPS_tif.csv'
        file = self.path + file_
        rain = pd.read_csv(file)

        year = np.arange(19) + 2000
        month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

        col_name = ['id']
        for yy in year:
            for mm in month:
                col_name.append(str(yy) + '.'+ mm + '._2')

        col_name = (np.asarray(col_name)[:-1]).tolist()
        rain_out = rain[col_name]
        rain_out['2018.12._2'] = rain_out['2017.12._2']

        new_col_name = ['id']
        for yy in year:
            for mm in month:
                new_col_name.append(str(yy) + '_'+ mm )

        rain_out.columns = new_col_name

        return rain_out


    def read_temp(self):

        #.ti is count, '_1' is sum , '_2' is mean which is what we want

        file_ = 'Temperature_subset.csv'
        file = self.path + file_
        temp = pd.read_csv(file)

        year = np.arange(19) + 2000
        month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

        col_name = ['id']
        for yy in year:
            for mm in month:
                col_name.append(str(yy) + '_'+ mm + '._2')

        col_name = (np.asarray(col_name)[:-1]).tolist()
        temp_out = temp[col_name]
        temp_out['2018_12._2'] = temp_out['2017_12._2']

        new_col_name = ['id']
        for yy in year:
            for mm in month:
                new_col_name.append(str(yy) + '_'+ mm)

        temp_out.columns = new_col_name

        return temp_out



    def get_nrisk_groups(self):

        file_ = 'Eth_AMAR_at_Risk_Groups_subset.csv'
        file =  self.path + file_
        atrisk = pd.read_csv(file)
        objid = np.asarray(atrisk['OBJECTID'])
        atriskid = np.asarray(atrisk['id'])

        el = len(self.id)
        nrisk = np.zeros(el)

        for i in np.arange(el):
            ind = atriskid == self.id[i]
            if ind.sum() > 0:
                nrisk[i] = len(np.unique(objid[ind]))

        return nrisk


    def read_GDP(self):

        #Set years > 2015 to 2015 value
        #the lack of data for years >2015 makes this data less useful

        file_ = 'GDP_PPP_1990_2015_5arcmin_v2_subset.csv'
        file = self.path + file_
        GDP = pd.read_csv(file)

        year = np.arange(15) + 2000

        col_name = ['id']
        for yy in year:
            col_name.append(str(yy) + '_mean')

        col_name = np.asarray(col_name).tolist()

        GDP_out = GDP[col_name]
        GDP_out['2015_mean'] = GDP['2015_mea_2']
        GDP_out['2016_mean'] = GDP['2015_mea_2']
        GDP_out['2017_mean'] = GDP['2015_mea_2']


        return GDP_out


    def read_gov_wgi_government_effectiveness(self):

        file_ = 'Gov_WGI_Governance_Quality_Government_Effectiveness.csv'
        file = self.path + file_
        gov = pd.read_csv(file)
        gov['yr_2001'] = (gov['yr_2000'] + gov['yr_2002'])/2
        gov['yr_2018'] = gov['yr_2017']

        year = np.arange(19) + 2000
        col_name = ['Country']
        for yy in year:
            col_name.append('yr_' + str(yy))

        gov_ext = gov[col_name]
        df_out = pd.DataFrame({'id':self.id, 'Country' : self.country})

        gov_merge = pd.merge(df_out, gov_ext, on='Country')
        gov_merge = gov_merge.sort_values('id')

        return gov_merge


    def read_gov_wgi_government_control_of_corruption(self):

        file_ = 'Gov_WGI_Governance_Quality_Control_of_Corruption.csv'
        file = self.path + file_
        gov = pd.read_csv(file)
        gov['yr_2001'] = (gov['yr_2000'] + gov['yr_2002'])/2
        gov['yr_2018'] = gov['yr_2017']

        year = np.arange(19) + 2000
        col_name = ['Country']
        for yy in year:
            col_name.append('yr_' + str(yy))

        gov_ext = gov[col_name]
        df_out = pd.DataFrame({'id':self.id, 'Country' : self.country})

        gov_merge = pd.merge(df_out, gov_ext, on='Country')
        gov_merge = gov_merge.sort_values('id')

        return gov_merge



    def read_gov_wgi_government_political_stability(self):

        file_ = 'Gov_WGI_Governance_Quality_Political_Stability_Absence_of_Violence.csv'
        file = self.path + file_
        gov = pd.read_csv(file)
        gov['yr_2001'] = (gov['yr_2000'] + gov['yr_2002'])/2
        gov['yr_2018'] = gov['yr_2017']

        year = np.arange(19) + 2000
        col_name = ['Country']
        for yy in year:
            col_name.append('yr_' + str(yy))

        gov_ext = gov[col_name]
        df_out = pd.DataFrame({'id':self.id, 'Country' : self.country})

        gov_merge = pd.merge(df_out, gov_ext, on='Country')
        gov_merge = gov_merge.sort_values('id')

        return gov_merge


    def read_gov_wgi_government_regulatory_quality(self):

        file_ = 'Gov_WGI_Governance_Quality_Regulatory_quality.csv'
        file = self.path + file_
        gov = pd.read_csv(file)
        gov['yr_2001'] = (gov['yr_2000'] + gov['yr_2002'])/2
        gov['yr_2018'] = gov['yr_2017']

        year = np.arange(19) + 2000
        col_name = ['Country']
        for yy in year:
            col_name.append('yr_' + str(yy))

        gov_ext = gov[col_name]
        df_out = pd.DataFrame({'id':self.id, 'Country' : self.country})

        gov_merge = pd.merge(df_out, gov_ext, on='Country')
        gov_merge = gov_merge.sort_values('id')

        return gov_merge



    def read_gov_wgi_government_rule_of_law(self):

        file_ = 'Gov_WGI_Governance_Quality_Rule_of_Law.csv'
        file = self.path + file_
        gov = pd.read_csv(file)
        gov['yr_2001'] = (gov['yr_2000'] + gov['yr_2002'])/2
        gov['yr_2018'] = gov['yr_2017']

        year = np.arange(19) + 2000
        col_name = ['Country']
        for yy in year:
            col_name.append('yr_' + str(yy))

        gov_ext = gov[col_name]
        df_out = pd.DataFrame({'id':self.id, 'Country' : self.country})

        gov_merge = pd.merge(df_out, gov_ext, on='Country')
        gov_merge = gov_merge.sort_values('id')

        return gov_merge


    def read_gov_wgi_government_voice_and_accountability(self):

        file_ = 'Gov_WGI_Governance_Quality_Voice_and_Accountability.csv'
        file = self.path + file_
        gov = pd.read_csv(file)
        gov['yr_2001'] = (gov['yr_2000'] + gov['yr_2002'])/2
        gov['yr_2018'] = gov['yr_2017']

        year = np.arange(19) + 2000
        col_name = ['Country']
        for yy in year:
            col_name.append('yr_' + str(yy))

        gov_ext = gov[col_name]
        df_out = pd.DataFrame({'id':self.id, 'Country' : self.country})

        gov_merge = pd.merge(df_out, gov_ext, on='Country')
        gov_merge = gov_merge.sort_values('id')

        return gov_merge


    def read_hybrid_food(self):

        file_ = 'Hybrid_10042015v9_subset.csv'
        file = self.path + file_
        food = pd.read_csv(file)

        return food['Food_mean']




    def get_nroads(self):

        #this just counts to total number of roads, 
        #should experiment with more robust quanitification
        
        file_ = 'groads_v1_global_subset.csv'
        file =  self.path + file_
        roads = pd.read_csv(file)

        nroads = []

        for i in self.id:
            nroads.append( (roads['id'] == i).sum() )

        return np.asarray(nroads)


    def get_npolitical_groups(self):

        #this just counts to total number of political groups, 
        #should experiment with more robust quanitification
        
        file_ = 'Eth_EPR_Politically_Relevant_subset.csv'
        file =  self.path + file_
        groups = pd.read_csv(file)

        ngroups = []

        for i in self.id:
            ngroups.append( (groups['id'] == i).sum() )


        return np.asarray(ngroups)



    def get_subnational_HDI(self):
        
        #HDI defined on subnational level using different amdinistrative boundaries,
        #there is not a one-to-one correspondence between vector overlay and output of 
        #HDI intersect. Thus the additional matching.

        file_ = 'GDL_Subnational_HDI_21_subset.csv'
        file =  self.path + file_
        HDI = pd.read_csv(file)

        HDI = HDI.loc[ (HDI['country'] != 'Belize') & (HDI['country'] != 'Mexico') & (HDI['country'] != 'Nicaragua') ]
        HDI = HDI.sort_values('id')
        HDI = HDI.reset_index(drop=True)

        el = len(self.id)
        good = np.zeros(el) - 1

        for i in np.arange(el):
            ind = np.where(HDI['id'] == self.id[i])[0]
            if len(ind) == 1: good[i] = ind
            elif len(ind) > 1:
                iii = ind[np.where(HDI['Shape_Area'].iloc[ind] == np.amax(HDI['Shape_Area'].iloc[ind]))[0]]
                good[i] = iii
            else:
                good[i] = good[i-1]


        HDI = HDI.iloc[good]
        HDI = HDI.reset_index(drop=True)

        year = np.arange(18) + 2000
        col_name = ['id']
        for yy in year:
            col_name.append('yr_' + str(yy))

        HDI_out = HDI[col_name]
        HDI_out['yr_2018'] = HDI_out['yr_2017']

        return HDI_out


    def temp_rain_feature_vector(self, pdrain, pdtemp, year_train):
        
        #temp and rain data at monthly resolution. Use difference and ratio for
        #temp and rain data, respectively, because want routine to see changes from
        #year to year. Currently, rain data not contributing significantly to regression.
        #Worth checking if taking rain data difference improves results.

        month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        col_name0 = [str(year_train[-1]) + '_' + mm for mm in month]

        rain_vec = [pdrain[col_name0].to_numpy()]
        temp_vec = [pdtemp[col_name0].to_numpy()]

        for yt in year_train[:-1]:
            col_name = [str(yt) + '_' + mm for mm in month]
            rain_vec.append(pdrain[col_name0].to_numpy() / pdrain[col_name].to_numpy())
            temp_vec.append(pdtemp[col_name0].to_numpy() - pdtemp[col_name].to_numpy())

        rain_out = np.hstack(rain_vec)
        rain_out[rain_out < 0] = np.median(rain_out[rain_out > 0])
        temp_out = np.hstack(temp_vec)

        return rain_out, temp_out 


    def gov_arms_hdi_feature_vector(self, in_vec, year_train):

        col_name0 = ['yr_' + str(yr) for yr in year_train]
        out_vec = in_vec[col_name0].to_numpy()

        return out_vec


    def gdp_feature_vector(self, gdp, year_train):

        col_name0 = [str(yr) + '_mean' for yr in year_train]
        gdp_vec = gdp[col_name0].to_numpy()
        gdp_vec[gdp_vec <= 0] = np.median(gdp_vec[gdp_vec > 0])
        
        return gdp_vec

    def pop_dp_feature_vector(self, pop, year_train):

        col_name_pop = [str(yr) + '_mean' for yr in year_train]
        pop_vec = pop[col_name_pop].to_numpy()

        col_name_dp = ['dp_' + str(yr) for yr in year_train]
        dp_vec = pop[col_name_dp].to_numpy()


        return pop_vec, dp_vec


    def bioclim_indices(self, rain, temp):
        
        #These features were engineered following the description at https://www.worldclim.org/bioclim.
        #Used temperature data even though it is temperature anomaly and not actualy temp measurements. 
        #worth checking if these definitions are appropriate.
        
        #couldn't produce these
        #BIO2 = Mean Diurnal Range (Mean of monthly (max temp - min temp))
        #BIO3 = Isothermality (BIO2/BIO7) (* 100)

        #BIO1 = Annual Mean Temperature
        BI01 = np.nanmean(temp, axis = 1)
        #BIO4 = Temperature Seasonality (standard deviation *100)
        BI04 = np.nanstd(temp, axis=1)
        #BIO5 = Max Temperature of Warmest Month
        BI05 = np.nanmax(temp, axis=1)
        #BIO6 = Min Temperature of Coldest Month
        BI06 = np.nanmin(temp, axis=1)
        #BIO7 = Temperature Annual Range (BIO5-BIO6)
        BI07 = BI05 - BI06
        #BIO12 = Annual Precipitation
        BI12 = np.nansum(rain, axis=1)
        #BIO13 = Precipitation of Wettest Month
        BI13 = np.nanmax(rain, axis=1)
        #BIO14 = Precipitation of Driest Month
        BI14 = np.nanmin(rain, axis=1)
        #BIO15 = Precipitation Seasonality (Coefficient of Variation)
        BI15 = np.nanstd(rain, axis=1)

        temp_quart = np.asarray([np.nanmean(temp[:,0+i:3+i], axis=1) for i in np.arange(9)]).T
        rain_quart = np.asarray([np.nansum(rain[:,0+i:3+i], axis=1) for i in np.arange(9)]).T

        ind_wettest = np.argmax(rain_quart, axis = 1)
        ind_driest = np.argmin(rain_quart, axis = 1)
        ind_warmest = np.argmax(temp_quart, axis=1)
        ind_coldest = np.argmin(temp_quart, axis=1)
        ind_dummy = np.arange(len(temp_quart[:,0]))

        #BIO8 = Mean Temperature of Wettest Quarter
        BI08 = temp_quart[ind_dummy, ind_wettest]
        #BIO9 = Mean Temperature of Driest Quarter
        BI09 = temp_quart[ind_dummy, ind_driest]
        #BIO10 = Mean Temperature of Warmest Quarter
        BI10 = temp_quart[ind_dummy, ind_warmest]
        #BIO11 = Mean Temperature of Coldest Quarter
        BI11 = temp_quart[ind_dummy, ind_coldest]
        #BIO16 = Precipitation of Wettest Quarter
        BI16 = rain_quart[ind_dummy, ind_wettest]
        #BIO17 = Precipitation of Driest Quarter
        BI17 = rain_quart[ind_dummy, ind_driest]
        #BIO18 = Precipitation of Warmest Quarter
        BI18 = rain_quart[ind_dummy, ind_warmest]
        #BIO19 = Precipitation of Coldest Quarter
        BI19 = rain_quart[ind_dummy, ind_coldest]

        bout = np.vstack([BI01, BI04, BI05, BI06, BI07, BI08, BI09, BI10, BI11, BI12, BI13, BI14, BI15, BI16, BI17, BI18, BI19]).T

        return bout

    
    def get_bioclim_data(self, pdrain, pdtemp, year_train):

        month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        bio = []
        for yt in year_train:
            col_name = [str(yt) + '_' + mm for mm in month]
            rain_vec = pdrain[col_name].to_numpy()
            temp_vec = pdtemp[col_name].to_numpy()
            bio.append(self.bioclim_indices(rain_vec, temp_vec)) 

        return np.hstack(bio)
    
    

    def get_training_data(self):

        #trying to predict, e.g. dp17
        #this is change in 2017 pop relative to 2016 pop
        #use nyears back data starting fron predicted year - 1
        #so for 2017 prediction and nyears = 3, use 2014, 2015, 2016


        #read in data time dependent data
        pop = self.read_population()
        arms = self.read_arms_exports()
        rain = self.read_rain()
        temp = self.read_temp()
        gdp = self.read_GDP()
        gov1 = self.read_gov_wgi_government_effectiveness()
        gov2 = self.read_gov_wgi_government_control_of_corruption()
        gov3 = self.read_gov_wgi_government_political_stability()
        gov4 = self.read_gov_wgi_government_regulatory_quality()
        gov5 = self.read_gov_wgi_government_rule_of_law()
        gov6 = self.read_gov_wgi_government_voice_and_accountability()
        hdi = self.get_subnational_HDI()

        #time independent data
        nrisk = self.get_nrisk_groups()
        food = self.read_hybrid_food()
        nroads = self.get_nroads()
        country = self.country
#        country_label_encoder = LabelEncoder()
#        country_integer_encoded = country_label_encoder.fit_transform(country)
#        country_onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
#        country_integer_encoded = country_integer_encoded.reshape(len(country_integer_encoded), 1)
#        country_onehot_encoded = country_onehot_encoder.fit_transform(country_integer_encoded)
        country_numeric = np.zeros(len(country))
        country_numeric[country == 'Guatemala'] = 1
        country_numeric[country == 'Honduras'] = 2
        feature_vec_time_ind = np.hstack(   [np.expand_dims(country_numeric, 1),
                                             np.expand_dims(nrisk, 1),
                                             np.expand_dims(food, 1),
                                             np.expand_dims(nroads, 1)]   )

        nyears = self.nyears_lookback
        year_pred = []
        pop_last_year = []
        label = []
        dplabel = []
        features = []

        el = 17 - nyears
        for i in np.arange(el):
            #constructe time dependence features
            yt = np.arange(nyears) + self.year_min + i
            #construct rain and temperature vector
            rain_vec, temp_vec = self.temp_rain_feature_vector(rain, temp, yt)
            rain_vec[rain_vec < 0] = np.median(rain_vec[rain_vec > 0])
            bio_vec = self.get_bioclim_data(rain, temp, yt)
            #construct gov vector
            gov_vec1 = self.gov_arms_hdi_feature_vector(gov1, yt)
            gov_vec2 = self.gov_arms_hdi_feature_vector(gov2, yt)
            gov_vec3 = self.gov_arms_hdi_feature_vector(gov3, yt)
            gov_vec4 = self.gov_arms_hdi_feature_vector(gov4, yt)
            gov_vec5 = self.gov_arms_hdi_feature_vector(gov5, yt)
            gov_vec6 = self.gov_arms_hdi_feature_vector(gov6, yt)
            gov_vec = np.hstack([gov_vec1, gov_vec2, gov_vec3, gov_vec4, gov_vec5, gov_vec6])
            #construct arms vector
            arms_vec = self.gov_arms_hdi_feature_vector(arms, yt)
            #construct hdi vector
            hdi_vec = self.gov_arms_hdi_feature_vector(hdi, yt)
            #construct gdp vector
            gdp_vec = self.gdp_feature_vector(gdp, yt)
            #construct pop and dp vector
            pop_vec, dp_vec = self.pop_dp_feature_vector(pop, yt)
            pop_vec[pop_vec <= 0] = np.median(pop_vec[pop_vec > 0] )
            dp_vec[np.isinf(dp_vec)] = np.median(dp_vec[np.isfinite(dp_vec)])
            dp_vec[dp_vec <= 0] = np.median(dp_vec[dp_vec > 0])
            
            col_name_id_lat_lon = ['left', 'top', 'id']
            lat_lon_id = np.asarray(pop[col_name_id_lat_lon])
            
            #construct feature vector
            fvec = np.hstack([feature_vec_time_ind, 
                              np.log10(rain_vec), 
                              temp_vec, 
                              bio_vec, 
                              gov_vec, 
                              arms_vec, 
                              hdi_vec, 
                              np.log10(gdp_vec), 
                              np.log10(pop_vec), 
                              np.log10(dp_vec), 
                              lat_lon_id ])
            if i == 0: features = np.asarray(fvec)
            else: features = np.vstack([features, fvec])

            #construct label
            yp = nyears + self.year_min + i
            year_pred.append(np.ones(8600)*yp)
            dplabel.append(pop['dp_' + str(yp)])
            label.append(pop[str(yp) + '_mean'])
            pop_last_year.append(pop[str(yp-1) + '_mean'])

            
        year_pred = np.asarray(year_pred).ravel()
        label = np.log10(np.asarray(label).ravel())
        dplabel = np.log10(np.asarray(dplabel).ravel())
        pop_last_year = np.log10(np.asarray(pop_last_year).ravel())
        
        fin_ind = np.isfinite(label) & np.isfinite(dplabel) & np.isfinite(pop_last_year) & (year_pred > 2001 + nyears) #2000 and 2001 pop data messed up
        label = label[fin_ind]
        dplabel = dplabel[fin_ind]
        features = features[fin_ind,:]
        year_pred = year_pred[fin_ind]
        pop_last_year = pop_last_year[fin_ind]

        
        ind_good = ~np.any(np.isnan(features), axis=1)
        label = label[ind_good]
        dplabel = dplabel[ind_good]
        features = features[ind_good, :]
        year_pred = year_pred[ind_good]
        pop_last_year = pop_last_year[ind_good]

        sel_ind = (np.abs(dplabel) > 0.1) & (dplabel > -1) & (dplabel < 1)
        label = label[sel_ind]
        dplabel = dplabel[sel_ind]
        features = features[sel_ind, :]
        year_pred = year_pred[sel_ind]
        pop_last_year = pop_last_year[sel_ind]        

        self.year_pred     = year_pred
        self.delta_rel_pop = dplabel
        self.pop_last_year = pop_last_year
        self.pop_this_year = label

        if nyears == 1:
            definition = np.asarray(['Country', 'N Groups at Risk', 'Food', 'N Roads', 
                                     np.repeat('Rain', 12), np.repeat('Temperature', 12), 
                                     np.repeat('BioClim', 17), np.repeat('Government Vector', 6), 
                                     np.repeat('Arms', 1), np.repeat('HDI', 1), np.repeat('GDP', 1), 
                                     np.repeat('Population', 1), np.repeat('Relative change in Population', 1), 
                                    'lat', 'lon', 'id'])
            definition = np.hstack(definition)            
        elif nyears == 3:
            #this is a quick and dirty cleaning of feature data which is only useful for 
            #the current setup. If, for example, nyears is changed, this cleanup step
            #will need to be revisited. For outliers, I take the median.
        
            features[features[:,87] < 0, 87] = np.median(features[features[:,87] > 0, 87])
            features[features[:,88] < 0, 88] = np.median(features[features[:,88] > 0, 88])
            features[features[:,89] < 0, 89] = np.median(features[features[:,89] > 0, 89])
            features[features[:,91] < 0, 91] = np.median(features[features[:,91] > 0, 91])
            features[features[:,92] < 0, 92] = np.median(features[features[:,92] > 0, 92])
            features[features[:,93] < 0, 93] = np.median(features[features[:,93] > 0, 93])
            features[features[:,94] < 0, 94] = np.median(features[features[:,94] > 0, 94])
            features[features[:,104] < 0, 104] = np.median(features[features[:,104] > 0, 104])
            features[features[:,105] < 0, 105] = np.median(features[features[:,105] > 0, 105])
            features[features[:,106] < 0, 106] = np.median(features[features[:,106] > 0, 106])
            features[features[:,108] < 0, 108] = np.median(features[features[:,108] > 0, 108])
            features[features[:,109] < 0, 109] = np.median(features[features[:,109] > 0, 109])
            features[features[:,110] < 0, 110] = np.median(features[features[:,110] > 0, 110])
            features[features[:,111] < 0, 111] = np.median(features[features[:,111] > 0, 111])
            features[features[:,121] < 0, 121] = np.median(features[features[:,121] > 0, 121])
            features[features[:,122] < 0, 122] = np.median(features[features[:,122] > 0, 122])
            features[features[:,123] < 0, 123] = np.median(features[features[:,123] > 0, 123])
            features[features[:,125] < 0, 125] = np.median(features[features[:,125] > 0, 125])
            features[features[:,126] < 0, 126] = np.median(features[features[:,126] > 0, 126])
            features[features[:,127] < 0, 127] = np.median(features[features[:,127] > 0, 127])
            features[features[:,128] < 0, 128] = np.median(features[features[:,128] > 0, 128])
        
            #this definition only applies to the current use of nyears = 3
            definition = np.asarray(['Country', 'N Groups at Risk', 'Food', 'N Roads', 
                                     np.repeat('Rain', 36), np.repeat('Temperature', 36), 
                                     np.repeat('BioClim', 51), np.repeat('Government Vector', 21), 
                                     np.repeat('Arms', 3), np.repeat('HDI', 3), np.repeat('GDP', 3), 
                                     np.repeat('Population', 3), np.repeat('Relative change in Population', 3), 
                                    'lat', 'lon', 'id'])
            definition = np.hstack(definition)
        else: definition = -1
        
        #output the features, the target labels to be fit and a crude definition of target labels.
        
        if self.classify:
            dplabel[dplabel<0] = 0
            dplabel[dplabel>0] = 1
        
        return features, dplabel, definition
