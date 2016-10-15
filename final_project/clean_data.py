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



