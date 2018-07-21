import sys,getopt,re
import os
import nltk
from collections import Counter
from sklearn.metrics import f1_score

############################splite sentence############################
def deal_with_sentence(read_document):##The first step is split word and label

	list_for_store_dealed_word=[]#store trained word
	list_for_dealed_label=[]#store trained label
	for i in read_document: ##deal with data, put them into tuple
		k=re.split('\n|\t| ',i)
		k.remove('')
		length=int(len(k)/2)
		word_list=k[0:length]
		label_list=k[length:]
		list_for_store_dealed_word.append(word_list)#store word of single sentence
		list_for_dealed_label.append(label_list)##store all label of single sentence

	return list_for_store_dealed_word,list_for_dealed_label#the first returned value is all the sentence word. The second returned value store all the label
#########################build feature space for current word-current label##############################
def build_feature_space_model_1(label_model_1,word_model_1,state):

	Phi_for_predict_model_1={s:{k:1 for k in word_model_1} for s in state}#build predict dictionary.Suppose all feature will be occured.
	Phi_for_correct_model_1={s:{k:0 for k in word_model_1} for s in state}#build actural dictionary.All feature are set as 0 initially.

	tuple_label_word_model_1=list(zip(label_model_1,word_model_1))#put single word and its corresponding label in a tuple in order to ease the statistic frequency of label-current word

	for i,j in Counter(tuple_label_word_model_1).items():
		for o,p in Phi_for_correct_model_1.items():
			for l,s in p.items():
				if i[0]==o and i[1]==l:
					Phi_for_correct_model_1[o][l]=j#if current word-current label occured, the value will be set as the frequence.
	
	return Phi_for_predict_model_1,Phi_for_correct_model_1

###################################build feature space for current word-current label and previous label-current label#############################
def build_feature_space_model_2(label_model_2,word_model_2,state): 
	
	word_model_2.extend(state)##add label target to all the sentence word in order to store all the possible label-word combination later in this function

	Phi_for_predict_model_2={s:{k:1 for k in word_model_2} for s in state} #Build a nested predict dictionary，the dictionary include all possible label-word and label-label feature. Suppose all feature will be occured and set the value as 1.
	Phi_for_predict_model_2.update({'Start':{z:1 for z in state}})#Because the first label need a 'Start' target. So update the dictionary, add 'Start'-label feature in the dictionary

	correct_Phi={s:{k:0 for k in word_model_2} for s in state}#Build actural dictionary.the dictionary also include all possible label-word and label-label feature. Set the value as 0 initially.
	correct_Phi.update({'Start':{z:0 for z in state}})#Update the dictionary, add 'Start'-label feature

	for i in state: #delete all the added label target in every sentence
		word_model_2.remove(i)

	tuple_label_word=list(zip(label_model_2,word_model_2))#put single word and its corresponding label in a tuple in order to ease the statistic frequency of label-current word

	list_feature_occur_model_2=[]
	for z in range(len(label_model_2)):##put all occurred label-label label-word combination, in order to statistic the frequency of label-label and label-word later in this function
		if z==0:
			tuple_1=('Start',label_model_2[z])
			list_feature_occur_model_2.append(tuple_1)
		else:
			tuple_2=(label_model_2[z-1],label_model_2[z])
			list_feature_occur_model_2.append(tuple_2)
	list_feature_occur_model_2.extend(tuple_label_word)

	for i,j in Counter(list_feature_occur_model_2).items(): #处理求得正确的Phi
		for o,p in correct_Phi.items():
			for l,s in p.items():
				if i[0]==o and i[1]==l:
					correct_Phi[o][l]=j#get the correct Phi for current word-current label and previous label-current label
	
	return Phi_for_predict_model_2,correct_Phi

#########################build feature space for previous words-current label##############################
def build_feature_space_model_3(label_model_3,word_model_3,state):

	word_model_3.append('Start')
	Phi_for_predict_model_3={s:{k:1 for k in word_model_3} for s in state}##Also, build the predict dictionary first.
	Phi_for_correct_model_3={s:{k:0 for k in word_model_3} for s in state}##Initialize the correct dictionary as 0 first.
	word_word_with_Start=[]
	for i in word_model_3:
		word_word_with_Start.append(i)
	for i in word_model_3:
		word_model_3.remove('Start')
		break
	Phi_for_update_predict={s:{k:1 for k in word_model_3} for s in word_word_with_Start}
	Phi_for_update_correct={s:{k:0 for k in word_model_3} for s in word_word_with_Start}

	Phi_for_predict_model_3.update(Phi_for_update_predict)
	Phi_for_correct_model_3.update(Phi_for_update_correct)

	tuple_label_word_model_3=list(zip(label_model_3,word_model_3))

	list_feature_occur_model_3=[]##Use the similiar method to statistic the frequency of feature.
	for z in range(len(word_model_3)):
		if z==0:
			tuple_model_3=('Start',word_model_3[z])
			list_feature_occur_model_3.append(tuple_model_3)
		else:
			tuple_model_3_1=(word_model_3[z-1],word_model_3[z])
			list_feature_occur_model_3.append(tuple_model_3_1)
	list_feature_occur_model_3.extend(tuple_label_word_model_3)
	
	for i,j in Counter(list_feature_occur_model_3).items(): 
		for o,p in Phi_for_correct_model_3.items():
			for l,s in p.items():
				if i[0]==o and i[1]==l:
					Phi_for_correct_model_3[o][l]=j
	
	return Phi_for_predict_model_3,Phi_for_correct_model_3
################################build w for current word-current label###################################
def build_w_model_1(word_list):
	all_word_model_1=[]
	for i in word_list:
		all_word_model_1.extend(i)
	for i in list_for_test_word:##put all word into the list including the test document word, in order to avoid the words in the test set that do not appear in the dictionary when doing the test
		all_word_model_1.extend(i)
	all_word_and_label_model=list(set(all_word_model_1))
	w_model_1={s:{z:0 for z in all_word_and_label_model} for s in state}##build the dictionary of w for all possible combination
	return w_model_1
################################build w for current word-current label and previous label-current label###################################
def build_w_model_2(word_list):
	all_word=[]
	for i in word_list:
		all_word.extend(i)
	for i in list_for_test_word:
		all_word.extend(i)
	all_word_and_label=list(set(all_word))
	all_word_and_label.extend(state)
	w_model_2={s:{z:0 for z in all_word_and_label} for s in state}
	w_model_2.update({'Start':{s:0 for s in state}})  #构建全局变量w 初始化全部为0

	return w_model_2
################################build w for  previous words-current word###################################
def build_w_model_3(word_list):
	all_word_model_3=[]
	word_word_list_without_Start=[]
	for i in word_list:
		all_word_model_3.extend(i)
		word_word_list_without_Start.extend(i)
	for i in list_for_test_word:
		all_word_model_3.extend(i)
		word_word_list_without_Start.extend(i)

	all_word_model_3.extend('Start')##add 'Start' target into all word in order to build feature space
	all_word_with_Start_label=list(set(all_word_model_3))
	all_word_without_Start_label=list(set(word_word_list_without_Start))
	w_model_3={s:{z:0 for z in all_word_with_Start_label} for s in state}#the labe corresponds to all the words and set as 0 initially

	w_for_update={s:{k:0 for k in all_word_without_Start_label} for s in all_word_with_Start_label}##build previous word-current word dictionary
	w_model_3.update(w_for_update)

	return w_model_3
################################Viterbi algorithm to find the maximum path of current word-current label###################################
def Viterbi_model_1(state,single_sentence_word_model1,Phi_model_1,w_model_1):

	curr_value={}
	for s in state:
		curr_value[s]=w_model_1[s][single_sentence_word_model1[0]]*Phi_model_1[s][single_sentence_word_model1[0]]
	
	path_model1 = {s:[] for s in state}
	for i in range(1,len(single_sentence_word_model1)):#use for loop to traverse the rest cur-label and cur-word
		last_value=curr_value
		curr_value={}
		for curr_state in state:
			max_value_of_all_path_model_1,last_state=max((last_value[last_state]+w_model_1[curr_state][single_sentence_word_model1[i]]*Phi_model_1[curr_state][single_sentence_word_model1[i]],last_state) for last_state in state)##find the maximum value of all possibel path and record the path

			curr_value[curr_state]=max_value_of_all_path_model_1
			path_model1[curr_state].append(last_state) 

	max_value_of_all_path_model_1=-1
	max_path_model_1=''
	for s in state:
		path_model1[s].append(s)#find the maximum path
		if curr_value[s]>max_value_of_all_path_model_1:
			max_value_of_all_path_model_1=curr_value[s]
			max_path_model_1=path_model1[s]

	return max_path_model_1

################################Viterbi method to find the maximum path of  word-current label and previous label-current label##################################
def  Viterbi_model_2(state,single_sentence_word,Phi_model_2,w_model_2):#Use Viterbi algorithm to find the maximum path
	
	curr_value={}
	for s in state:
		curr_value[s]=w_model_2['Start'][s]*Phi_model_2['Start'][s]*w_model_2[s][single_sentence_word[0]]*Phi_model_2[s][single_sentence_word[0]]##record all the first 'Start'-label and label-word value

	path_record= {s:[] for s in state}##this state not include 'Start' label
	for i in range(1,len(single_sentence_word)):#use for loop to traverse the rest previous label-label and label-word tags
		last_value=curr_value
		curr_value={}
		for curr_state in state:#Use curr_value to record current state 
			max_value_of_all_path_model_2,last_state=max((last_value[last_state]+w_model_2[last_state][curr_state]*Phi_model_2[last_state][curr_state]*w_model_2[curr_state][single_sentence_word[i]]*Phi_model_2[curr_state][single_sentence_word[i]],last_state)for last_state in state)
			##record the last label path that got the maximum score
			curr_value[curr_state]=max_value_of_all_path_model_2
			path_record[curr_state].append(last_state) ##record all possible path
	max_value_of_all_path_model_2=-1
	max_path=''
	for s in state:#find the maximum path
		path_record[s].append(s)
		if curr_value[s]>max_value_of_all_path_model_2:
			max_value_of_all_path_model_2=curr_value[s]
			max_path=path_record[s]
	return max_path

################################Viterbi method to find the maximum path of previous words-current word###################################
def  Viterbi_model_3(state,single_sentence_word_model3,Phi_model_3,w_model_3):
	
	curr_pro={}	
	for s in state:
		curr_pro[s]=w_model_3['Start'][single_sentence_word_model3[0]]*Phi_model_3['Start'][single_sentence_word_model3[0]]*w_model_3[s][single_sentence_word_model3[0]]*Phi_model_3[s][single_sentence_word_model3[0]]

	path = {s:[] for s in state}
	for i in range(1,len(single_sentence_word_model3)):##use for loop to traverse the rest previous words-current word
		last_pro=curr_pro
		curr_pro={}
		for curr_state in state:#
			max_value_of_all_path_model_3,last_state=max((last_pro[last_state]+w_model_3[curr_state][single_sentence_word_model3[i]]*Phi_model_3[curr_state][single_sentence_word_model3[i]]*w_model_3[single_sentence_word_model3[i-1]][single_sentence_word_model3[i]]*Phi_model_3[single_sentence_word_model3[i-1]][single_sentence_word_model3[i]],
				last_state) for last_state in state)##record the last label path that got the maximum score
			curr_pro[curr_state]=max_value_of_all_path_model_3
			path[curr_state].append(last_state) ##record all possible path
	max_value_of_all_path_model_3=-1
	max_path_model_3=''
	for s in state:#find the maximum path
		path[s].append(s)
		if curr_pro[s]>max_value_of_all_path_model_3:
			max_value_of_all_path_model_3=curr_pro[s]
			max_path_model_3=path[s]

	return max_path_model_3

##############################update the w for current word-current label########################################

def update_w_model_1(sentence_word_model1,predict_label_model1,dct_for_correct_Phi_model1,correct_label_model_1,w_model_1):
	for i in range(len(sentence_word_model1)):
		if correct_label_model_1[i]!=predict_label_model1[i]:
			w_model_1[correct_label_model_1[i]][sentence_word_model1[i]]=w_model_1[correct_label_model_1[i]][sentence_word_model1[i]]+dct_for_correct_Phi_model1[correct_label_model_1[i]][sentence_word_model1[i]]##Increase the value of w for the correct feature
			w_model_1[predict_label_model1[i]][sentence_word_model1[i]]=w_model_1[predict_label_model1[i]][sentence_word_model1[i]]-dct_for_correct_Phi_model1[predict_label_model1[i]][sentence_word_model1[i]]#decrease the value of w for the incorrect feature
	return w_model_1

###############################update the w for current word-current label and previous label-current label########################################

def update_w_model_2(sentence_word_model_2,dct_for_predict_Phi,dct_for_correct_Phi_model_2,w_model_2,state,predict_labelmodel_2,correct_label_model_2):
	
	for i in range(len(sentence_word_model_2)):
		if correct_label_model_2[i]!=predict_labelmodel_2[i]:#if the predict result is incorrect
			if i>0:#update transfer probability 
				w_model_2[correct_label_model_2[i-1]][correct_label_model_2[i]]=w_model_2[correct_label_model_2[i-1]][correct_label_model_2[i]]+dct_for_correct_Phi_model_2[correct_label_model_2[i-1]][correct_label_model_2[i]]#更新状态转移概率
				w_model_2[predict_labelmodel_2[i-1]][predict_labelmodel_2[i]]=w_model_2[predict_labelmodel_2[i-1]][predict_labelmodel_2[i]]-dct_for_correct_Phi_model_2[predict_labelmodel_2[i-1]][predict_labelmodel_2[i]]
			else:#update the start label feature 
				w_model_2['Start'][correct_label_model_2[i]]=w_model_2['Start'][correct_label_model_2[i]]+dct_for_correct_Phi_model_2['Start'][correct_label_model_2[i]]
				w_model_2['Start'][predict_labelmodel_2[i]]=w_model_2['Start'][predict_labelmodel_2[i]]-dct_for_correct_Phi_model_2['Start'][predict_labelmodel_2[i]]

			w_model_2[correct_label_model_2[i]][sentence_word_model_2[i]]=w_model_2[correct_label_model_2[i]][sentence_word_model_2[i]]+dct_for_correct_Phi_model_2[correct_label_model_2[i]][sentence_word_model_2[i]]##update the label-word feature
			w_model_2[predict_labelmodel_2[i]][sentence_word_model_2[i]]=w_model_2[predict_labelmodel_2[i]][sentence_word_model_2[i]]-dct_for_correct_Phi_model_2[predict_labelmodel_2[i]][sentence_word_model_2[i]]
		
	return w_model_2

###############################update the w for previous words-current word########################################
def update_w_model_3(sentence_word_model_3,predict_label_model_3,dct_for_correct_Phi_model_3,correct_label_model_3,w_model_3):
	for i in range(len(sentence_word_model_3)):
		if correct_label_model_3[i]!=predict_label_model_3[i]:
			if i>0:
				w_model_3[sentence_word_model_3[i-1]][sentence_word_model_3[i]]=w_model_3[sentence_word_model_3[i-1]][sentence_word_model_3[i]]+dct_for_correct_Phi_model_3[sentence_word_model_3[i-1]][sentence_word_model_3[i]]
				w_model_3[sentence_word_model_3[i]][sentence_word_model_3[i-1]]=w_model_3[sentence_word_model_3[i]][sentence_word_model_3[i-1]]-dct_for_correct_Phi_model_3[sentence_word_model_3[i]][sentence_word_model_3[i-1]]
			else:
				w_model_3['Start'][sentence_word_model_3[i]]+=dct_for_correct_Phi_model_3['Start'][sentence_word_model_3[i]]

			w_model_3[correct_label_model_3[i]][sentence_word_model_3[i]]=w_model_3[correct_label_model_3[i]][sentence_word_model_3[i]]+dct_for_correct_Phi_model_3[correct_label_model_3[i]][sentence_word_model_3[i]]
			w_model_3[predict_label_model_3[i]][sentence_word_model_3[i]]=w_model_3[predict_label_model_3[i]][sentence_word_model_3[i]]-dct_for_correct_Phi_model_3[predict_label_model_3[i]][sentence_word_model_3[i]]
			
	return w_model_3

##############################doing the test########################################
def doing_test(list_for_test_word,state,w_model_1,w_model_2,w_model_3):
	list_for_predict_result_model2=[]
	list_for_predict_result_model1=[]
	list_for_predict_result_model3=[]
	count=0
	for i in list_for_test_word:
		feature_model_1=build_feature_space_model_1(list_for_test_label[count],i,state)
		get_predict_path_model1=Viterbi_model_1(state,i,feature_model_1[1],w_model_1)#using the predict feature and value of w to do the predict for the result label 

		feature_model_2=build_feature_space_model_2(list_for_test_label[count],i,state)#using the predict feature and value of w to do the predict for the result label 
		get_predict_path=Viterbi_model_2(state,i,feature_model_2[1],w_model_2)

		feature_model_3=build_feature_space_model_3(list_for_test_label[count],i,state)#using the predict feature and value of w to do the predict for the result label 
		get_path_predict=Viterbi_model_3(state,i,feature_model_3[1],w_model_3)

		list_for_predict_result_model2.extend(get_predict_path)
		list_for_predict_result_model1.extend(get_predict_path_model1)
		list_for_predict_result_model3.extend(get_path_predict)
		count=count+1

	list_for_correct_label=[]
	for i in list_for_test_label:
		list_for_correct_label.extend(i)

	f1_micro_for_model1 = f1_score(list_for_correct_label, list_for_predict_result_model1, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
	f1_micro_for_model2 = f1_score(list_for_correct_label, list_for_predict_result_model2, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
	f1_micro_for_model3 = f1_score(list_for_correct_label, list_for_predict_result_model3, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
	
	print('')
	print('f1 value of current word-current label is: ',f1_micro_for_model1)
	sort_dct_model_1={}
	for k,j in w_model_1.items():
		for z,l in j.items():
			tuple_for_model_1=(k,z)
			sort_dct_model_1.update({tuple_for_model_1:w_model_1[k][z]})
	sorted_dct_model_1=sorted(sort_dct_model_1.items(),key=lambda k:k[1],reverse=True)
	output_model_1={}
	for k in sorted_dct_model_1[0:10]:
		update={k[0]:k[1]}
		output_model_1.update(update)
	print("the top 10 most positive features for current word-current label is:",output_model_1)

	print('')

	print('f1 value of current word-current label and previous label-current label is: ',f1_micro_for_model2)
	sort_dct_model_2={}
	for k,j in w_model_2.items():
		for z,l in j.items():
			tuple_for_model_2=(k,z)
			sort_dct_model_2.update({tuple_for_model_2:w_model_2[k][z]})
	sorted_dct_model_2=sorted(sort_dct_model_2.items(),key=lambda k:k[1],reverse=True)
	output_model_2={}
	for k in sorted_dct_model_2[0:10]:
		update={k[0]:k[1]}
		output_model_2.update(update)
	print("the top 10 most positive features for current word-current label and previous label-current label is:",output_model_2)

	print('')

	print('f1 value of previous words-current word is: ',f1_micro_for_model3)
	sort_dct_model_3={}
	for k,j in w_model_3.items():
		for z,l in j.items():
			tuple_for_model_3=(k,z)
			sort_dct_model_3.update({tuple_for_model_3:w_model_3[k][z]})
	sorted_dct_model_3=sorted(sort_dct_model_3.items(),key=lambda k:k[1],reverse=True)
	output_model_3={}
	for k in sorted_dct_model_3[0:10]:
		update={k[0]:k[1]}
		output_model_3.update(update)
	print("the top 10 most positive features for previous words-current word is:",output_model_3)
##################################main方法###################################
if __name__ == '__main__':
	str_1=sys.argv[1]
	read_train_document=open(str_1,'r+')

	str_2=sys.argv[2]
	read_test_document=open(str_2,'r+')

	get_dealed_sentence=deal_with_sentence(read_train_document)
	word_list=get_dealed_sentence[0]##The word list for training document
	label_list=get_dealed_sentence[1]##The label list for training document

	get_dealed_test_sentence=deal_with_sentence(read_test_document)
	list_for_test_word=get_dealed_test_sentence[0] ##word list for test document
	list_for_test_label=get_dealed_test_sentence[1]##label list for test document

	state_list=[]
	for i in label_list:
		state_list.extend(i)
	state=list(set(state_list))##统计一共出现多少次的隐含状态

	list_for_predict_Phi_model_1=[]
	list_for_correct_Phi_model_1=[]
	
	list_for_predict_Phi_model_2=[]
	list_for_correct_Phi_model_2=[]

	list_for_predict_Phi_model_3=[]
	list_for_correct_Phi_model_3=[]
	
	count=0
	for i in word_list: 
		feature_space_model_1=build_feature_space_model_1(label_list[count],i,state)
		feature_space_model_2=build_feature_space_model_2(label_list[count],i,state)
		feature_space_model_3=build_feature_space_model_3(label_list[count],i,state)

		list_for_predict_Phi_model_1.append(feature_space_model_1[0])
		list_for_correct_Phi_model_1.append(feature_space_model_1[1])

		list_for_predict_Phi_model_2.append(feature_space_model_2[0])
		list_for_correct_Phi_model_2.append(feature_space_model_2[1])

		list_for_predict_Phi_model_3.append(feature_space_model_3[0])
		list_for_correct_Phi_model_3.append(feature_space_model_3[1])

		count=count+1
########################################buil the w for current word-current label########################################
	w_model_1=build_w_model_1(word_list)
########################################buil the w for current word-current label and previous label-current label###############################
	w_model_2=build_w_model_2(word_list)
########################################buil the w for previous words-current word###############################
	w_model_3=build_w_model_3(word_list)
##########################################update the w########################################
	count=0
	for i in word_list:##deal with the training document
		predict_path_model_1=Viterbi_model_1(state,i,list_for_predict_Phi_model_1[count],w_model_1)###get the max path for current word-current label
		predict_path_model_2=Viterbi_model_2(state,i,list_for_predict_Phi_model_2[count],w_model_2)##get the max path for current word-current label and previous label-current label
		predict_path_model_3=Viterbi_model_3(state,i,list_for_predict_Phi_model_3[count],w_model_3)#get the max path for  previous words-current word

		feature_model_2=build_feature_space_model_2(predict_path_model_2,i,state)
		Phi_for_predict_path=feature_model_2[1]
		w_model_2=update_w_model_2(i,Phi_for_predict_path,list_for_correct_Phi_model_2[count],w_model_2,state,predict_path_model_2,label_list[count])
		w_model_1=update_w_model_1(i,predict_path_model_1,list_for_correct_Phi_model_1[count],label_list[count],w_model_1)
		w_model_3=update_w_model_3(i,predict_path_model_3,list_for_correct_Phi_model_3[count],label_list[count],w_model_3)
		count=count+1
##########################################进行test###############################################
	test=doing_test(list_for_test_word,state,w_model_1,w_model_2,w_model_3)






