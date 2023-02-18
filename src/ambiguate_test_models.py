
import sys
import os
import json
import logging
import numpy as np

from tqdm import tqdm
from google.protobuf.json_format import MessageToDict

class Model:
    """
    형태소 분석기 API 모델을 모아놓은 객체
    비교할 수 있게 각기 다른 품사를 세종 태그셋으로 변환한다.
    """
    @staticmethod
    def from_name(name, args=None):
        if name == 'kiwi': return KiwiModel()
        if name == 'komoran': return KomoranModel()
        if name == 'kkma': return KkmaModel()
        if name == 'hannanum': return HannanumModel()
        if name == 'mecab': return MecabModel()
        if name == 'khaiii': return KhaiiiModel()
        if name == 'bareun': return BareunModel(args.bareun_api_key) if args else EtriModel()
        if name == 'etri': return EtriModel(args.etri_api_key) if args else EtriModel()

    def _convert(self, morph):
        raise NotImplementedError()

    def _tokenize(self, text):
        raise NotImplementedError()

    def tokenize(self, text):
        if type(self).__name__ == "HannanumModel":
            anal_tokens = []
            tokens = self._tokenize(text)
            for token in tokens:
                if len(token) == 0:
                    continue
                if len(token) == 1:
                    for tt in token:
                        for t in tt:
                            _t = self._convert(t)
                            anal_tokens.append(_t)
                else:
                    for tt in token[0]:
                        _t = self._convert(tt)
                        anal_tokens.append(_t)
            return anal_tokens
        elif type(self).__name__ == "KhaiiiModel":
            anal_tokens = []
            analyzed = self._tokenize(text)
            for word in analyzed:
                for morph in word.morphs:
                    anal_tokens.append((morph.lex, morph.tag))
            return anal_tokens
        elif type(self).__name__ == "BareunModel":
            anal_tokens = []
            analyzed = self._tokenize(text)
            for word in analyzed['sentences'][0]['tokens']:
                for morph in word['morphemes']:
                    anal_tokens.append((morph['text']['content'], morph['tag']))
            return anal_tokens
        elif type(self).__name__ == "EtriModel":
            anal_tokens = []
            analyzed = self._tokenize(text)
            for token in analyzed['return_object']['sentence'][0]['morp']:
                anal_tokens.append((token['lemma'], token['type']))
            return anal_tokens
        elif type(self).__name__ == "KiwiModel":
            anal_tokens = []
            analyzed = self._tokenize(text)
            for word in analyzed:
                form = word.form
                tag = word.tag
                if word.tag == 'VV-I' or word.tag == 'VV-R':
                    tag = 'VV'
                elif word.tag == 'VA-I' or word.tag == 'VA-R':
                    tag = 'VA'
                elif "ᆯ" in word.form:
                    form = word.form.replace("ᆯ", "ㄹ")
                elif "ᆫ" in word.form:
                    form = word.form.replace("ᆫ", "ㄴ")
                elif "ᆸ" in word.form:
                    form = word.form.replace("ᆸ", "ㅂ")
                elif "ᆷ" in word.form:
                    form = word.form.replace("ᆷ", "ㅁ")
                anal_tokens.append((form, tag))
            return anal_tokens
        else:
            return list(map(self._convert, self._tokenize(text)))

class EtriModel(Model):
    def __init__(self, api_key=None):
        import urllib3
        print("Initialize Etri API", file=sys.stderr)
        self.accessKey = api_key if api_key else "DEFAULT_API_KEY" # Etri 홈페이지에서 받은 key or a default key
        self.analysisCode = "morp"
        self.openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
        self.http = urllib3.PoolManager()

    def _tokenize(self, text):
        headers = {}
        headers['Content-Type'] = "application/json; charset=UTF-8"
        headers['Authorization'] = self.accessKey

        requestJson = {
                "argument": {
                    "text": text,
                    "analysis_code": self.analysisCode
                }
            }
        res = self.http.request(
            "POST",
            self.openApiURL,
            headers=headers,
            body=json.dumps(requestJson)
        )
        res2 = json.loads(res.data)
        return res2

class BareunModel(Model):
    def __init__(self, api_key):
        import bareunpy
        from bareunpy import Tagger
        print("Initialize bareunpy ({})".format(bareunpy.bareun_version), file=sys.stderr)
        self.tagger = Tagger(api_key, 'localhost')

    def _tokenize(self, text):
        return MessageToDict(self.tagger.tags([text]).msg())


class KiwiModel(Model):
    def __init__(self):
        import kiwipiepy
        from kiwipiepy import Kiwi
        print("Initialize kiwipiepy ({})".format(kiwipiepy.__version__), file=sys.stderr)
        self._mdl = Kiwi()

    def _convert(self, morph):
        return morph.form, morph.tag

    def _tokenize(self, text):
        return self._mdl.tokenize(text)

class KomoranModel(Model):
    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Komoran from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Komoran()

    def _convert(self, morph):
        return morph[0], morph[1]

    def _tokenize(self, text):
        return self._mdl.pos(text)

class KkmaModel(Model):
    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Kkma from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Kkma()

    def _convert(self, morph):
        if morph[1] == 'NNM':
            return morph[0], 'NNB'
        if morph[1] == 'VXV' or \
           morph[1] == 'VXA':
            return morph[0], 'VX'
        if morph[1] == 'MDT':
            return morph[0], 'MMA'
        if morph[1] == 'MDN':
            return morph[0], 'MMN'
        if morph[1] == 'MAC':
            return morph[0], 'MAJ'
        if morph[1] == 'JKI':
            return morph[0], 'JKV'
        if morph[1] == 'JKM':
            return morph[0], 'JKB'
        if morph[1] == 'EPH' or\
           morph[1] == 'EPT' or\
           morph[1] == 'EPP': # 존칭, 시제, 공손 선어말어미
            return morph[0], 'EP'
        if morph[1] == 'EFN' or\
           morph[1] == 'EFQ' or\
           morph[1] == 'EFO' or\
           morph[1] == 'EFA' or\
           morph[1] == 'EFI' or\
           morph[1] == 'EFR': # 평서, 의문, 명령, 청유, 감탄, 존칭 종결어미
            return morph[0], 'EF'
        if morph[1] == 'ECE' or\
           morph[1] == 'ECD' or\
           morph[1] == 'ECS': # 대등, 의존, 보조적 연결어미
            return morph[0], 'EC'
        if morph[1] == 'ETD':
            return morph[0], 'ETM'
        if morph[1] == 'UN': # 명사 추정 범주
            return morph[0], 'NF'
        if morph[1] == 'OL': 
            return morph[0], 'SL'
        if morph[1] == 'OH': 
            return morph[0], 'SH'
        if morph[1] == 'ON': 
            return morph[0], 'SN'

        return morph[0], morph[1]

    def _tokenize(self, text):
        return self._mdl.pos(text)

class MecabModel(Model):
    def __init__(self):
        import konlpy
        from konlpy.tag import Mecab
        print("Initialize Mecab from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = Mecab()

    def _convert(self, morph):
        # (morph[1][:2] if morph[1].startswith('V') else morph[1][:1])
        return morph[0], morph[1]

    def _tokenize(self, text):
        return self._mdl.pos(text)

class HannanumModel(Model):

    def __init__(self):
        import konlpy
        from konlpy import tag
        print("Initialize Hannanum from konlpy ({})".format(konlpy.__version__), file=sys.stderr)
        self._mdl = tag.Hannanum()

    def _convert(self, morph):
        if morph[1] == 'ncpa' or\
           morph[1] == 'ncps' or\
           morph[1] == 'ncn' or\
           morph[1] == 'ncr':
            return morph[0], 'NNG'
        if morph[1] == 'nqpa' or\
           morph[1] == 'nqpb' or\
           morph[1] == 'nqpc' or\
           morph[1] == 'nqq':
            return morph[0], 'NNP'
        if morph[1] == 'nbu' or\
           morph[1] == 'nbs' or\
           morph[1] == 'nbn':
            return morph[0], 'NNB'
        if morph[1] == 'npp' or\
           morph[1] == 'npd':
            return morph[0], 'NP'
        if morph[1] == 'nnc' or\
           morph[1] == 'nno':
            return morph[0], 'NR'
        if morph[1] == 'pvd' or\
           morph[1] == 'pvg':
            return morph[0], 'VV'
        if morph[1] == 'pad' or\
           morph[1] == 'paa':
            return morph[0], 'VA'
        if morph[1] == 'px':
            return morph[0], 'VX'
        if morph[1] == 'mmd':
            return morph[0], 'MMD'
        if morph[1] == 'mma':
            return morph[0], 'MMA'
        if morph[1] == 'mad' or\
           morph[1] == 'mag':
            return morph[0], 'MAG'
        if morph[1] == 'maj':
            return morph[0], 'MAJ'
        if morph[1] == 'ii':
            return morph[0], 'IC'
        if morph[1] == 'jcs':
            return morph[0], 'JKS'
        if morph[1] == 'jcc':
            return morph[0], 'JKC'
        if morph[1] == 'jcv':
            return morph[0], 'JKV'
        if morph[1] == 'jcj' or\
           morph[1] == 'jct':
            return morph[0], 'JC'
        if morph[1] == 'jcr':
            return morph[0], 'JKQ'
        if morph[1] == 'jco':
            return morph[0], 'JKO'
        if morph[1] == 'jca':
            return morph[0], 'JKB'
        if morph[1] == 'jcm':
            return morph[0], 'JKG'
        if morph[1] == 'jxc' or\
           morph[1] == 'jxcn' or\
           morph[1] == 'jxf':
            return morph[0], 'JX'
        if morph[1] == 'jp':
            return morph[0], 'VCP'
        if morph[1] == 'ecc' or\
           morph[1] == 'ecx' or\
           morph[1] == 'ecs':
            return morph[0], 'EC'
        if morph[1] == 'etn':
            return morph[0], 'ETN'
        if morph[1] == 'etm':
            return morph[0], 'ETM'
        if morph[1] == 'ef':
            return morph[0], 'EF'
        if morph[1] == 'ep':
            return morph[0], 'EP'
        if morph[1] == 'xp':
            return morph[0], 'XPN'
        if morph[1] == 'xsnu' or\
           morph[1] == 'xsna' or\
           morph[1] == 'xsnca' or\
           morph[1] == 'xsns' or\
           morph[1] == 'xsncc' or\
           morph[1] == 'xsnp' or\
           morph[1] == 'xsnx':
            return morph[0], 'XSN'
        if morph[1] == 'xsvv' or\
           morph[1] == 'xsva' or\
           morph[1] == 'xsvn':
            return morph[0], 'XSV'
        if morph[1] == 'xsms' or\
           morph[1] == 'xsam' or\
           morph[1] == 'xsas' or\
           morph[1] == 'xsa' or\
           morph[1] == 'xsmn':
            return morph[0], 'XSA'
        if morph[1] == 'sp':
            return morph[0], 'SP'
        if morph[1] == 'sf':
            return morph[0], 'SF'
        if morph[1] == 'sl' or\
           morph[1] == 'sr':
            return morph[0], 'SS'
        if morph[1] == 'sy':
            return morph[0], 'SW'
        if morph[1] == 'f':
            return morph[0], 'SL'
        if morph[1] == 'se':
            return morph[0], 'SE'
        if morph[1] == 'sd':
            return morph[0], 'SO'

        return morph[0], morph[1]

    def _tokenize(self, text):
        return self._mdl.analyze(text)

class KhaiiiModel(Model):
    def __init__(self):
        from khaiii import KhaiiiApi
        print("Initialize Khaiii ({})".format("0.4"), file=sys.stderr)
        api = KhaiiiApi()

        self.mdl = api

    def _tokenize(self, text):
        return self.mdl.analyze(text)

def file_path(dir_path):
    """
    평가 데이터 경로를 불러온다.
    """
    final_path = []
    for (root, dir, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            final_path.append(file_path)
    final_path.sort()
    return final_path


def load_dataset(json_list):
    """
    평가 데이터를 비교할 수 있게 토큰 단위로 전처리한다.
    """
    ret = []
    
    for file in json_list:
        with open(file, 'r') as f:
            data = json.load(f)
        for text in data:
            pre = []
            arr = np.array(data[0]['split_answer'])
            if len(arr.shape) == 1:
                pre.append((text['split_answer'][0], text['split_answer'][1]))
                pre.append(text['text'])
            else:
                pre.append(text['split_answer'])
                pre.append(text['text'])
            ret.append(pre)
    return ret

def evaluate(dataset, model, error_output=None, print_all_results=False):
    """
    모델별로 모호성 평가 실행
    :param dataset: 평가 데이터셋
    :param model: 평가 모델
    :param error_output: 오류 출력
    :param print_all_result: 전부 출력
    :return : 정확도, 오답 log
    """
    acc, tot = 0, 0
    print(f'testing {type(model).__name__}')
    for _, (answer, exam) in enumerate(tqdm(dataset)):
        correct = False
        try:
            result = model.tokenize(exam)
            if type(model).__name__ == "KhaiiiModel":
                for i, t in enumerate(result):
                    # khaiii 모델은 관형사를 mm으로만 분류하므로 그냥 세부분류까지 맞춘것으로 본다.
                    if ("MM" in t[1] and "MM" in answer[1]) and (t[0] == answer[0]):
                        result[i] = (answer[0], answer[1])
        except:
            print(f'{type(model).__name__} tokenizer error!')
            continue
        
        answer_length = len(answer)
        correct_cnt = 0
        for idx, token in enumerate(result):
            # 첫번째 토큰이 정답이면 정답의 길이만큼 다음 토큰을 계산한다.
            i = 0
            if isinstance(answer, list): # 정답이 두개 이상의 토큰이 경우(ex. [(쿠키, NNG), (와, JC)])
                if (token[0] == answer[i][0]) and (token[1] == answer[i][1]):
                    correct_cnt += 1
                    while i <= answer_length:
                        i += 1
                        if i < answer_length:
                            try:
                                if (result[idx+i][0] == answer[i][0]) and (result[idx+i][1] == answer[i][1]):
                                    correct_cnt +=1
                                    if correct_cnt == answer_length:
                                        correct = True
                                        break
                            except: # 정답이 결과보다 많은 경우는 분석을 잘못한 경우이므로 오답처리
                                correct = False
                                break
            else: # 정답이 하나의 토큰인 경우
                if (token[0] == answer[0]) and (token[1] == answer[1]):
                    correct = True

        acc += int(correct)
        tot += 1
        if (print_all_results or not correct) and error_output is not None:
            print(answer, ':\n-->', *map('/'.join, result), file=error_output)
        logger.info(f'{type(model).__name__}')
        logger.info(f'acc {acc}/ total {tot} = {acc/tot:.4f}')
    return acc / tot

def main(args):
    """
    모호성 해소 정확도 테스트

    """
    model_names = args.target.split(',')
    models =  [Model.from_name(n, args) if n == 'etri' or n =='bareun' else Model.from_name(n) for n in model_names]

    if args.error_output_dir:
        os.makedirs(args.error_output_dir, exist_ok=True)
        error_outputs = [open(args.error_output_dir + '/' + name + '.error.txt', 'w', encoding='utf-8') for name in model_names]
    else:
        error_outputs = None
    sys.stdout = open('output/accuracy_ambiguity.txt', 'w')
    print('', *model_names, sep='\t')
    datasets = file_path(args.path)
    ds = load_dataset(datasets)
    scores = []
    for i, model in enumerate(models):
        if type(model).__name__ == "EtriModel":
            ds = ds[:5000]
        acc = evaluate(ds, model, error_output=(error_outputs[i] if error_outputs else None), print_all_results=False) 
        scores.append(acc)

    print('acc', *((f'{s:.3f}' if s is not None else '-') for s in scores), sep='\t')
    sys.stdout.close()
    if error_outputs:
        for f in error_outputs:
            f.close()
    sys.stdout = sys.__stdout__

def get_logger(dir, filename):
    dirs = dir.split("/")
    filename = f'/{filename}'
    if not os.path.exists(dirs[0]):
        os.makedirs(dirs[0])
    if not os.path.exists(dir):
        os.makedirs(dir)
    if os.path.exists(dir+filename):
        os.remove(dir+filename)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(dir+filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger = logging.getLogger(dir+filename.split('.')[0])
    if not len(logger.handlers) > 0:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.propagate = False

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes._base').disabled = True

    return logger

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='path' , help='read from data(json)', type=str)
    parser.add_argument('-t', dest='target', help='bareun,komoran,mecab,kkma,hannanum,kiwi,etri', type=str)
    parser.add_argument('-etri', dest='etri_api_key', help='For EtriModel, an API key is required.', type=str)
    parser.add_argument('-bareun', dest='bareun_api_key', help='For BareunModel, an API key is required.', type=str)
    parser.add_argument('-o', dest='error_output_dir', help='error output')
    parser.add_argument('--print_all_results', default=False, action='store_true')

    logger = get_logger("output/log", "ambiguity_accuracy.log")
    
    main(parser.parse_args()) # 모호성 정확도 평가 실행
    print('done !! Please refer to the file "output/accuracy_ambiguity.txt"')

