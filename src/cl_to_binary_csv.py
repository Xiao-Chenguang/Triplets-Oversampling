import os
import pandas as pd
from scipy.io import arff

root = 'datasets/'

# **Tasks**:
# - convert to csv file
# - convert to 2 classes
# - remove header
# - save without index


def process_vehicle():
    print('==================== vehicle ====================')
    path_vehicle = root + 'raw/vehicle.arff'
    data_vehicle, meta_vehicle = arff.loadarff(path_vehicle)
    df_vehicle = pd.DataFrame(data_vehicle)
    df_vehicle['Class'].value_counts()

    # convert to csv with 2 classes: van and not van
    df_vehicle.replace(b'van', 1, inplace=True)
    df_vehicle.replace(b'saab', 0, inplace=True)
    df_vehicle.replace(b'bus', 0, inplace=True)
    df_vehicle.replace(b'opel', 0, inplace=True)
    df_vehicle.to_csv(root + 'vehicle.csv', index=False, header=False)


def process_diabete():
    print('==================== diabete ====================')
    path_diabetes = root + 'raw/diabete.arff'
    data_diabetes, meta_diabetes = arff.loadarff(path_diabetes)
    df_diabetes = pd.DataFrame(data_diabetes)
    # print(df_diabetes['Outcome'].value_counts())

    # convert to csv with 2 classes: van and not van
    df_diabetes.replace(b'1', 1, inplace=True)
    df_diabetes.replace(b'0', 0, inplace=True)
    df_diabetes.to_csv(root + 'diabete.csv', index=False, header=False)


def process_vowel():
    print('==================== vowel ====================')
    path_vowel = root + 'raw/vowel.arff'
    data_vowel, meta_vowel = arff.loadarff(path_vowel)
    # print(meta_vowel)
    df_vowel = pd.DataFrame(data_vowel)
    df_vowel = df_vowel[df_vowel.columns[2:]]

    vowel_class = {b'hid': 0, b'hId': 1, b'hEd': 2, b'hAd': 3, b'hYd': 4,
                   b'had': 5, b'hOd': 6, b'hod': 7, b'hUd': 8, b'hud': 9, b'hed': 10}

    # convert to csv with 2 classes: van and not van
    for k, v in vowel_class.items():
        if k == b'hid':
            df_vowel.replace(k, 1, inplace=True)
        else:
            df_vowel.replace(k, 0, inplace=True)
    df_vowel.to_csv(root + 'vowel.csv', index=False, header=False)


def process_ionosphere():
    print('==================== ionosphere ====================')
    path_ionosphere = root + 'raw/ionosphere.arff'
    data_ionosphere, meta_ionosphere = arff.loadarff(path_ionosphere)
    df_ionosphere = pd.DataFrame(data_ionosphere)
    df_ionosphere.replace(b'g', 0, inplace=True)
    df_ionosphere.replace(b'b', 1, inplace=True)
    # check column 1 has same value
    df_ionosphere.iloc[:, 1].value_counts()
    # drop columns 1 if it has same value
    df_ionosphere.drop(df_ionosphere.columns[1], axis=1, inplace=True)
    df_ionosphere.to_csv(root + 'ionosphere.csv',
                         index=False, header=False)


def process_abalone():
    print('==================== abalone ====================')
    path_abalone = root + 'raw/abalone.arff'
    data_abalone, meta_abalone = arff.loadarff(path_abalone)
    # 9 pos
    # 18 neg
    df_abalone = pd.DataFrame(data_abalone)
    df_abalone = df_abalone[df_abalone.columns[1:]]
    # print(df_abalone.head())
    df_abalone = df_abalone[(df_abalone['Class_number_of_rings'] == b'9')
                            | (df_abalone['Class_number_of_rings'] == b'18')]
    df_abalone.replace(b'9', 0, inplace=True)
    df_abalone.replace(b'18', 1, inplace=True)
    # print(df_abalone['Class_number_of_rings'].value_counts())
    df_abalone.to_csv(root + 'abalone.csv',
                      index=False, header=False)


def process_staimage():
    print('==================== staimage ====================')
    path_staimage = root + 'raw/satimage.arff'
    data_staimage, meta_staimage = arff.loadarff(path_staimage)
    df_staimage = pd.DataFrame(data_staimage)
    t = df_staimage['class'].value_counts()

    # convert to csv with 2 classes: van and not van
    df_staimage.replace(b'4.', 1, inplace=True)
    df_staimage.replace(b'1.', 0, inplace=True)
    df_staimage.replace(b'2.', 0, inplace=True)
    df_staimage.replace(b'5.', 0, inplace=True)
    df_staimage.replace(b'7.', 0, inplace=True)
    df_staimage.replace(b'3.', 0, inplace=True)
    df_staimage.to_csv(root + 'satimage.csv',
                       index=False, header=False)


def process_haberman():
    print('==================== haberman ====================')
    path_haberman = root + 'raw/haberman.arff'
    data_haberman, meta_haberman = arff.loadarff(path_haberman)
    df_haberman = pd.DataFrame(data_haberman)
    df_haberman["Patients_year_of_operation"] = df_haberman["Patients_year_of_operation"].str.decode('utf-8').astype(int)
    t = df_haberman['Survival_status'].value_counts()
    # print(t)

    #     # convert to csv with 2 classes: van and not van
    df_haberman.replace(b'2', 1, inplace=True)
    df_haberman.replace(b'1', 0, inplace=True)
    df_haberman.to_csv(root + 'haberman.csv',
                       index=False, header=False)


def process_aloi():
    print('==================== aloi ====================')
    path_pulsar = root + 'raw/aloi.arff'
    data_pulsar, meta_pulsar = arff.loadarff(path_pulsar)
    df_pulsar = pd.DataFrame(data_pulsar)
    df_pulsar = df_pulsar[~df_pulsar.duplicated()]

    t = df_pulsar['Target'].value_counts()
    # print(t)

    #     # convert to csv with 2 classes: van and not van
    df_pulsar.replace(b'Anomaly', 1, inplace=True)
    df_pulsar.replace(b'Normal', 0, inplace=True)
    df_pulsar.to_csv(root + 'aloi.csv',
                     index=False, header=False)


def process_pulsar():
    print('==================== pulsar ====================')
    path_pulsar = root + 'raw/pulsar.arff'
    data_pulsar, meta_pulsar = arff.loadarff(path_pulsar)
    df_pulsar = pd.DataFrame(data_pulsar)

    t = df_pulsar['0'].value_counts()
    # print(t)
    df_pulsar.to_csv(root + 'pulsar.csv', index=False, header=False)


def convert_to_csv():
    if os.path.exists(root + 'vehicle.csv'):
        print('vehicle.csv already exists')
    else :
        process_vehicle()
    if os.path.exists(root + 'diabete.csv'):
        print('diabete.csv already exists')
    else :
        process_diabete()
    if os.path.exists(root + 'vowel.csv'):
        print('vowel.csv already exists')
    else :
        process_vowel()
    if os.path.exists(root + 'ionosphere.csv'):
        print('ionosphere.csv already exists')
    else :
        process_ionosphere()
    if os.path.exists(root + 'abalone.csv'):
        print('abalone.csv already exists')
    else :
        process_abalone()
    if os.path.exists(root + 'satimage.csv'):
        print('satimage.csv already exists')
    else :
        process_staimage()
    if os.path.exists(root + 'haberman.csv'):
        print('haberman.csv already exists')
    else :
        process_haberman()
    if os.path.exists(root + 'aloi.csv'):
        print('aloi.csv already exists')
    else :
        process_aloi()
    if os.path.exists(root + 'pulsar.csv'):
        print('pulsar.csv already exists')
    else :
        process_pulsar()

if __name__ == '__main__':
    convert_to_csv()
