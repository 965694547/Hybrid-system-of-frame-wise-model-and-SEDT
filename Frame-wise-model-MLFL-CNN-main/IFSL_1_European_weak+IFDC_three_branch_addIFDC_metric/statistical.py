import numpy as np
import re
import pandas as pd
import os


file_dir = "./results/"

def st_dev(value_list):
	return np.std(np.array(value_list), ddof=1)

valid_audio_taggings = []

#audio taggings
print ("Results of audio tagging")
print ("")
with open(os.path.join(file_dir, "valid_taggings.out")) as f:
	find_float = lambda x: re.search("\d+(\.\d+)?",x).group()
	for line in f:
		f1 = float(find_float(line.split(":")[1]))
		valid_audio_taggings += [f1]

mean_valid_audio_tagging_f1 = round(sum(valid_audio_taggings) / len(valid_audio_taggings), 3)
max_valid_audio_tagging_f1 = round(max(valid_audio_taggings), 3)
std_valid_audio_tagging_f1 = round(st_dev(valid_audio_taggings), 4)
max_valid_audio_tagging_f1_index = valid_audio_taggings.index(max(valid_audio_taggings))



print("mean_valid_audio_tagging_f1:", mean_valid_audio_tagging_f1,"(",std_valid_audio_tagging_f1, ")", max_valid_audio_tagging_f1)
print("std_valid_audio_tagging_f1:", std_valid_audio_tagging_f1)
print ("max_valid_audio_tagging_f1:", max_valid_audio_tagging_f1 )		
print ("")


audio_taggings = []

with open(os.path.join(file_dir, "taggings.out")) as f:
	find_float = lambda x: re.search("\d+(\.\d+)?",x).group()
	for line in f:
		f1 = float(find_float(line.split(":")[1]))
		audio_taggings += [f1]
audio_taggings.sort(reverse = True)
mean_audio_tagging_f1 = round(sum(audio_taggings) / len(audio_taggings), 3)
max_audio_tagging_f1 = round(max(audio_taggings), 3)
max_audio_tagging_f1_choose = round(audio_taggings[max_valid_audio_tagging_f1_index], 3)
std_audio_tagging_f1 = round(st_dev(audio_taggings), 4)
second_max_audio_tagging = audio_taggings[1]
print("mean_audio_tagging_f1:", mean_audio_tagging_f1, "(", std_audio_tagging_f1, ")", max_audio_tagging_f1)
print("std_audio_tagging_f1:", std_audio_tagging_f1)
print ("max_audio_tagging_f1:", max_audio_tagging_f1 )		
print ("max_audio_tagging_f1_choose:", max_audio_tagging_f1_choose )		
print ("second_max_audio_tagging", second_max_audio_tagging)
print ("")

tagging_series = pd.Series([(mean_audio_tagging_f1, std_audio_tagging_f1, max_audio_tagging_f1_choose)],index= ["f1"], name = "audio_tagging")

#segment results
print ("Results of segement:")
print ("")
valid_segments = []
with open(os.path.join(file_dir, "valid_segments.out")) as f:
	find_float = lambda x: re.search("\d+(\.\d+)?",x).group()
	for line in f:
		f1 = float(find_float(line.split(":")[1]))
		valid_segments += [f1]
mean_valid_segment_f1 = round(sum(valid_segments) / len(valid_segments), 3)
max_valid_segment_f1 = round(max(valid_segments), 3)
std_valid_segment_f1 = round(st_dev(valid_segments), 4)
max_valid_segment_f1_index = valid_segments.index(max(valid_segments))

print("mean_valid_segment_f1:", mean_valid_segment_f1,"(", std_valid_segment_f1,")", max_valid_segment_f1)
print("std_valid_segment_f1:", std_valid_segment_f1)
print ("max_valid_segment_f1:", max_valid_segment_f1 )		
print ("")


segments = []
with open(os.path.join(file_dir, "segments.out")) as f:
	find_float = lambda x: re.search("\d+(\.\d+)?",x).group()
	for line in f:
		f1 = float(find_float(line.split(":")[1]))
		segments += [f1]
segments.sort(reverse=True)		
mean_segment_f1 = round(sum(segments) / len(segments), 3)
max_segment_f1 = round(max(segments), 3)
std_segment_f1 = round(st_dev(segments), 4)
max_segment_f1_choose = round(segments[max_valid_segment_f1_index], 3)
second_max_segment_f1 = segments[1]
print("mean_segment_f1:", mean_segment_f1, "(",std_segment_f1, ")", max_segment_f1)
print("std_segment_f1:", std_segment_f1)
print ("max_segment_f1:", max_segment_f1 )
print ("max_segment_f1_choose", max_segment_f1_choose)
print ("")

segment_series = pd.Series([(mean_segment_f1, std_segment_f1, max_segment_f1_choose)], index = ["f1"], name = "segment")

# valid_event-based results
print ("Results of event")
print ("")
valid_events = []
with open(os.path.join(file_dir, "valid_events.out")) as f:
	find_float = lambda x: re.search("\d+(\.\d+)?",x).group()
	for line in f:
		f1 = float(find_float(line.split(":")[1]))
		valid_events += [f1]
mean_valid_event_f1 = round(sum(valid_events) / len(valid_events), 3)
max_valid_event_f1 = round(max(valid_events), 3)
std_valid_event_f1 = round(st_dev(valid_events), 4)
max_valid_event_f1_index = valid_events.index(max(valid_events))

print("mean_valid_event_f1:", mean_valid_event_f1, "(", std_valid_event_f1,")", max_valid_event_f1)
print("std_valid_event_f1:", std_valid_event_f1)
print ("max_valid_event_f1:", max_valid_event_f1 )		
print ("")


events = []
with open(os.path.join(file_dir, "events.out")) as f:
	find_float = lambda x: re.search("\d+(\.\d+)?",x).group()
	for line in f:
		f1 = float(find_float(line.split(":")[1]))
		events += [f1]
events.sort(reverse = True)		
mean_event_f1 = round(sum(events) / len(events), 3)
max_event_f1 = round(max(events), 3)
std_event_f1 = round(st_dev(events), 4)
max_event_f1_choose = round(events[max_valid_event_f1_index], 3)
second_max_event_f1 = events[1]

print("mean_event_f1:", mean_event_f1, "(", std_event_f1, ")", max_event_f1)
print("std_event_f1:", std_event_f1)
print ("max_event_f1:", max_event_f1 )		
print ("max_event_f1_choose", max_event_f1_choose)
print ("second_max_event_f1", second_max_event_f1)
print ("")

event_series = pd.Series([(mean_event_f1, std_event_f1, max_event_f1_choose)], index = ["f1"], name = "event")

result_df = pd.DataFrame({tagging_series.name:tagging_series, segment_series.name:segment_series, event_series.name:event_series})

print(result_df)

