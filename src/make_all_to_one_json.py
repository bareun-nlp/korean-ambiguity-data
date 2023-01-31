import json
import os


def file_path(dir_path):
    """
    평가 데이터 리스트를 불러온다.
    :param dir_path: 파일이 있는 경로
    :return final_path: 파일 경로 리스트
    """
    final_path = []
    for (root, _, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            final_path.append(file_path)
    final_path.sort()

    return final_path

def make_all_to_one_json(json_list):
    """
    각각의 폴더에 있는 평가 데이터를 하나로 합친다.
    :param json_list: 평가 데이터 리스트
    :return json: 하나로 합친 json파일
    """
    all_json = []
    for file in json_list:
        with open(file, 'r') as f:
            data = json.load(f)
        for text in data:
            all_json.append(text)
    with open('ambiguity_sentences_merge.json', 'w') as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2)
    print('succeeded in merging the json files into one.')
    print('Please see the ambiguity_sentences_merge.json file')


if __name__ == '__main__':
    json_list  = file_path("data")
    make_all_to_one_json(json_list)

    