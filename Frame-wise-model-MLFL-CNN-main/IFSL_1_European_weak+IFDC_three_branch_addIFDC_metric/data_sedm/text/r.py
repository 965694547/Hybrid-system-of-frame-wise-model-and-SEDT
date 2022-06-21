import csv
f = open('./soundscapes2021.csv')
lines = f.readlines()
f2 = open('./soundscapes2021_da.csv','w')
# reader = csv.DictReader(f)
if __name__ =='__main__':
    # print(lines)
    for rows in lines:
        f2.writelines('new_'+rows)
    #     print (lines)
f.close()
f2.close()
# for rows in f