# This python file is used to pre-process conversation,dark,phone charge,phone lock,bluetooth,wifi location,flourishing Score and Panas score.
import pandas as pd
import numpy as np
import os
import time
import re
from datetime import datetime
pd.set_option('display.width',200)

def get_file(path):
    file = [f for f in os.listdir(path) if re.search('(.+\.csv$)', f)]
    file = sorted(file)
    return file
path1 = "F://9417/StudentLife_Dataset/StudentLife_Dataset/Inputs/sensing/conversation"
path2 = "F://9417/StudentLife_Dataset/StudentLife_Dataset/Inputs/sensing/dark"
path3 = "F://9417/StudentLife_Dataset/StudentLife_Dataset/Inputs/sensing/phonecharge"
path4 = "F://9417/StudentLife_Dataset/StudentLife_Dataset/Inputs/sensing/phonelock"

uid = ['u00', 'u01', 'u02', 'u03', 'u04', 'u05', 'u07', 'u08', 'u09', 'u10',
       'u12', 'u13', 'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u22',
       'u23', 'u24', 'u25', 'u27', 'u30', 'u31', 'u32', 'u33', 'u34', 'u35',
       'u36', 'u39', 'u41', 'u42', 'u43', 'u44', 'u45', 'u46', 'u47', 'u49',
       'u50', 'u51', 'u52', 'u53', 'u54', 'u56', 'u57', 'u58', 'u59']

# -------------------conversation,dark,phone charge,phone lock--------------------#
# conversation, dark, phone charge and phone lock all are related to time duration
# so we pre-process them together
# in this part, we get how much time each student spend per day in these activities.
def times_duration(path,name,user):
    f_names = get_file(path)
    df_empty = pd.DataFrame(columns=[name])
    
    for i in range(len(f_names)):
        obj = pd.read_csv(path +'/' +f_names[i])
        col_list = obj.columns.values.tolist()
        #change the timestamp to real time
        obj[col_list[0]] = obj[col_list[0]].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        obj[col_list[1]] = obj[col_list[1]].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        obj['duration'] = obj.apply(lambda row: datetime.strptime(row[col_list[1]], '%Y-%m-%d %H:%M:%S') - datetime.strptime(row[col_list[0]], '%Y-%m-%d %H:%M:%S'), axis=1)
        #get the duration
        obj['date'] = obj[col_list[1]].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date())
        sum1 = obj['duration'].sum()
        days = pd.to_datetime(obj['date'].iloc[-1]) - pd.to_datetime(obj['date'].iloc[0])
        if(days.days != 0):
            time_per_day = sum1/days.days
            df_empty.loc[user[i]] = round(time_per_day.total_seconds()/3600,3)
            #change the day to seconds, which is more intuitive    
        else:
            df_empty.loc[user[i]] = round(sum1.total_seconds()/3600,3)   
    return df_empty


df_total = pd.concat([times_duration(path1,"conversation",uid),times_duration(path2,"dark",uid),times_duration(path3,"phonecharge",uid),times_duration(path4,"phonelock",uid)],axis=1)
df_total.to_csv("./df_total.csv")

# -------------------flourishing Score--------------------------------------------#
# because some student did not do all the questions
# so how many questions each student did and get the average score for each student


def flourishingScale(path):
    #f_names = get_file(path)
    obj = pd.read_csv(path)
    df_empty = pd.DataFrame(columns=["pre","post"])
    group_by_type=obj.groupby('type')
    obj_pre = group_by_type.get_group("pre")
    obj_post = group_by_type.get_group("post")
    for column in list(obj_pre.columns[obj_pre.isnull().sum() > 0]):
        #mean_val = round(obj_pre[column].mean(),3)
        obj_pre[column].fillna(0, inplace=True)
    obj_pre.set_index('uid',inplace=True)
    obj_pre.drop('type',axis=1, inplace=True)
    # get the number of questions the student did
    obj_pre["count"] = obj_pre.apply(lambda x : 8-x.value_counts().get(0,0),axis=1)
    obj_pre["sum"] = obj_pre.apply(lambda x: x.sum(), axis=1)

    #obj_pre.to_csv("./flouringshing_pre.csv")
 
    for column in list(obj_post.columns[obj_post.isnull().sum() > 0]):
        #mean_val = round(obj_post[column].mean(),3)
        obj_post[column].fillna(0, inplace=True)
    
    obj_post.set_index('uid',inplace=True)
    obj_post.drop('type',axis=1, inplace=True)
    obj_post["count"] = obj_post.apply(lambda x : 8-x.value_counts().get(0,0),axis=1)
    obj_post["sum"] = obj_post.apply(lambda x: x.sum(), axis=1)
    
    df_empty["pre"] =  obj_pre.apply(lambda x: (x["sum"]/x["count"])*8, axis=1)
    df_empty["post"] = obj_post.apply(lambda x: (x["sum"]/x["count"])*8, axis=1)
    return df_empty

flourishingScale("F://9417/StudentLife_Dataset/StudentLife_Dataset/Outputs/FlourishingScale.csv")
# ----------------------------------panasScale--------------------------------------------#
# from the specification, we seperate the file to positive one and negative one
# for each one, we did the same operation with flourishingScale
# get the average score for each student
def panasScale(path,uid):
    obj = pd.read_csv(path)
    #print(obj)
    positive = [1,4,8,9,11,12,14,15,17] #index of positive questions
    negative = [2,3,5,6,7,10,13,16,18]  #index of negative questions
    df_empty = pd.DataFrame(columns=["panas_preP","panas_preN","panas_postP","panas_postN"])
    group_by_type=obj.groupby('type')
    group_by_type=obj.groupby('type')
    obj_pre = group_by_type.get_group("pre")
    obj_pre.set_index("uid",inplace=True)
    obj_pre.drop('type',axis=1, inplace=True)
    
    obj_post = group_by_type.get_group("post")
    obj_post.set_index("uid",inplace=True)
    obj_post.drop('type',axis=1, inplace=True)
    obj_pre.columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    obj_post.columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    obj_pre_positive = obj_pre.ix[:,positive] 
    
    for column in list(obj_pre_positive.columns[obj_pre_positive.isnull().sum() > 0]):
        obj_pre_positive[column].fillna(0, inplace=True)#if is null, fill the null with 0
    obj_pre_positive["count"] = obj_pre_positive.apply(lambda x : 9-x.value_counts().get(0,0),axis=1)
    obj_pre_positive["sum"] = obj_pre_positive.apply(lambda x: x.sum(), axis=1)

    obj_pre_negative = obj_pre.ix[:,negative]
    for column in list(obj_pre_negative.columns[obj_pre_negative.isnull().sum() > 0]):
        obj_pre_negative[column].fillna(0, inplace=True)
    obj_pre_negative["count"] = obj_pre_negative.apply(lambda x : 9-x.value_counts().get(0,0),axis=1)
    obj_pre_negative["sum"] = obj_pre_negative.apply(lambda x: x.sum(), axis=1)
    print(obj_pre_negative)

    obj_post_positive = obj_post.ix[:,positive] 
    for column in list(obj_post_positive.columns[obj_post_positive.isnull().sum() > 0]):
        obj_post_positive[column].fillna(0, inplace=True)
    
    obj_post_positive["count"] = obj_post_positive.apply(lambda x : 9-x.value_counts().get(0,0),axis=1)
    obj_post_positive["sum"] = obj_post_positive.apply(lambda x: x.sum(), axis=1)
    obj_post_negative = obj_post.ix[:,negative]
    for column in list(obj_post_negative.columns[obj_post_negative.isnull().sum() > 0]):
        obj_post_negative[column].fillna(0, inplace=True)
    obj_post_negative["count"] = obj_post_negative.apply(lambda x : 9-x.value_counts().get(0,0),axis=1)
    obj_post_negative["sum"] = obj_post_negative.apply(lambda x: x.sum(), axis=1)
    

    df_empty["panas_preP"] = obj_pre_positive.apply(lambda x: ((x["sum"]-x["count"])/x["count"])*9, axis=1)
    df_empty["panas_preN"] = obj_pre_negative.apply(lambda x: ((x["sum"]-x["count"])/x["count"])*9, axis=1)
    df_empty["panas_postP"] = obj_post_positive.apply(lambda x: ((x["sum"]-x["count"])/x["count"])*9, axis=1)
    df_empty["panas_postN"] = obj_post_negative.apply(lambda x: ((x["sum"]-x["count"])/x["count"])*9, axis=1)
    df_empty.to_csv("./panas_processing.csv")
    
panasScale("F://9417/StudentLife_Dataset/StudentLife_Dataset/Outputs/panas.csv",uid)

# ----------------------------------BlueTooth------------------------------------------- #
# we get how many different devices a student connected a day
# which means how many people this student met a day
# this is related to people's social situation
def bluetooth(path,uid):
    f_names = get_file(path)
    df_bluetooth = pd.DataFrame(columns=["average"])
    for j in range(len(f_names)):
        i = f_names[j]
        obj = pd.read_csv(path + '/' + i)
        obj.drop('MAC',axis=1, inplace=True)
        # we do not need to use MAC
        # because class_id already tell us whether it is a same device
        obj.set_index('time',inplace=True)
        indexlist = obj.index.tolist()
        time_list = sorted(set(indexlist),key=indexlist.index)
        #print(time_list)
        df_empty = pd.DataFrame(columns=["date","class_list"])
        for i in range(len(time_list)):
            time = time_list[i]
            time_YMD = datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
            time_Y = datetime.strptime(time_YMD, '%Y-%m-%d %H:%M:%S').date()
            # change the timestamp to real time
            list2 = []
            list1 = obj.loc[time]["class_id"].tolist()
            
            if isinstance(list1,int):
                list2.append(list1)
            else:
                list2 = list1
            class_list = sorted(set(list2),key=list2.index)
            
            df_empty.loc[i,"class_list"] = class_list
            df_empty.loc[i,"date"] = time_Y
        #group all the device in the same day together and drop the duplicate devices
        date_list = df_empty["date"].tolist()
        time_list = []
        time_dict = {}
        for i in range(len(date_list)):
            date = date_list[i]
            if date in time_list:
                time_dict[date].extend(df_empty.loc[i,"class_list"])
            else:
                time_list.append(date)
                time_dict[date] = list(df_empty.loc[i,"class_list"])
        
        total_class = 0
        for i in time_dict.keys():
            values = sorted(set(time_dict[i]),key=time_dict[i].index)
            total_class += len(values)
        # get the total number of devices a day
        average = total_class/len(time_list) # get the average devices each student connected per day
         
bluetooth("F://9417/StudentLife_Dataset/StudentLife_Dataset/Inputs/sensing/gps",uid)

# ---------------------------------wifi location---------------------------------------------#
# we get the location manually using the map and divide them to four parts like below
label_dict = {
"study" :['35centerra','woodburyhall','wentworth','websterhall','vail','thornton','thayer_secure','Tuck_hall','sudikoff','baker-berry','steele','silsby-rocky','sanborn','ropeferry','rollins-chapel','buchanan','robinson','remsen','reed','carpenterhall','carson-tech_services','raven-house','chasehall','parkhurst','cummings','currier','murdough','dana-library','dartmouth_hall','dewey','moore','mcnutt','maclean','library-default-services','kemeny','kellogg','fairchild','hanoverpsych','feldberg_library','french'],
"daily_activity" : ['woodward','53_commons','whittemore','Cohen','wheeler','HanoverInn','topliff','streeter','ripley','richardson','newhamp','mclaughlin','bissell','maxwell','brown_hall','lord','lodge','butterfield','byrnehall','little_hall','judge','channing-cox','hallgarten','gile','fayerweather','fahey-mclane','cutter-north'],
"others" : ['tllc-raether','tllc','7-lebanon','remote_offices_HREAP','DCCCC','north-main','Mckenzie','lsb','isr_wireless','hitchcock','external_25lebanon','external','college-street','burke'],
"recreation" : ['vac','sport-venues-press','sport-venues','sphinx','softballfield','smith','presidents_house','occum','aquinas','north-park','batrlett','berry_sports_center','massrow','blunt_alumni_center','hopkins','hillcrest','fairbanks','evergreen','east-wheelock']}

# in this function, we get the Proportion of occuring in which part
def wifi_location(path,uid,label_dict):
    f_names = get_file(path)
    df_empty = pd.DataFrame(columns=["study","daily_activity","others","recreation"])
    for j in range(len(f_names)):
        count_dict = {"study":0,"daily_activity":0,"others":0,"recreation":0}
        i = f_names[j]
        wifi = []
        obj = pd.read_csv(path + '/' + i)
        location_list = obj["time"].tolist()
        for location in location_list:
            n = location.split('[')[0]
            if n == 'in': 
                wifi.append(location[3:-1])
        for i in wifi:
            for m in label_dict:
                if i in label_dict[m]:
                    count_dict[m] += 1
        total = 0
        for count in count_dict:
            total += count_dict[count]
        #print(count_dict["study"]/total)
        df_empty.loc[uid[j],"study"] = round(count_dict["study"]/total,3)
        df_empty.loc[uid[j],"daily_activity"] = round(count_dict["daily_activity"]/total,3)
        df_empty.loc[uid[j],"others"] = round(count_dict["others"]/total,3)
        df_empty.loc[uid[j],"recreation"] = round(count_dict["recreation"]/total,3)
    df_empty.to_csv("./wifi_location_processing.csv")
    return df_empty
    
wifi_location("F://9417/StudentLife_Dataset/StudentLife_Dataset/Inputs/sensing/wifi_location",uid,label_dict)