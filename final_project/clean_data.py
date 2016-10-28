import sys
from matplotlib import pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def clean_data(data_dict, feature):
    for each in data_dict:
        for ft in data_dict[each].keys():
            if ft == feature:
                if data_dict[each][ft] == 'NaN':
                    data_dict[each][ft] = 0
    return data_dict

def find_missing_values_ratio(data_dict):
    ratio = {}
    for i in data_dict.keys():
        features = data_dict[i].keys()
        break

    for i in range(0, len(features)):
        k = 0
        for j in data_dict.keys():
            if data_dict[j][features[i]] == 'NaN':
                k += 1
        ratio[features[i]] = round(k / float(len(data_dict)), 4)
    return ratio


def draw_scatter(data_dict, xy, desc):
    data = featureFormat(data_dict, xy)
    plt.figure()
    for point in data:
        salary = point[0]
        bonus = point[1]
        plt.scatter(salary, bonus, alpha=0.5)

    plt.title('Salary vs. Bonus')
    plt.xlabel('Salary')
    plt.ylabel('Bonus')
    plt.savefig(desc)



