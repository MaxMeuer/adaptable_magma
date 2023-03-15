# %%
from magma import Magma
from magma.image_input import ImageInput
from tqdm import tqdm
import json
import os
import argparse
from collections import Counter
import random
from rtpt import RTPT
from eval_utils import EvalHelper, aa_focused

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--config_path', type=str, default='configs/okvqa.yml')
parser.add_argument('--datasets', type=str, default='gqa,vqa,okvqa,vizwiz')
parser.add_argument('--dataset_path', type=str,
                    default='/storage-01/ml-mmeuer/datasets')
parser.add_argument('--coco_path', type=str,
                    default='/storage-01/ml-mmeuer/datasets/coco')
parser.add_argument('--temp', type=float, default=0.01)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--max_steps', type=int, default=6)
parser.add_argument('--num_samples', type=int, default=None)  # None for all
parser.add_argument('--aa_focused', type=bool, default=False)
parser.add_argument('--few_shot', type=int, default=0)

# %%


def get_coco_path(dir, id, mode=None, year=2014):
    assert year in [2014, 2017]
    if year == 2014:
        return os.path.join(dir, f'COCO_{mode}2014_{id:012}.jpg')
    elif year == 2014:
        return os.path.join(dir, f'{id:012}.jpg')


def create_prompt(question, answer=None):
    return f'Question: {question} Answer:' if answer is None else f'Question: {question} Answer: {answer}'


class Evaluator:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.eval_helper = EvalHelper()

    def generate(self, inputs):
        embeddings = model.preprocess_inputs(inputs).cuda()
        answer = self.model.generate(
            embeddings=embeddings,
            max_steps=self.args.max_steps,
            temperature=self.args.temp,
            top_k=self.args.top_k,
        )[0]
        return answer

    def clean_answer(self, answer, longest_answer=None):
        answer = answer.replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = answer.strip()
        answer = self.eval_helper.processPunctuation(answer)
        answer = self.eval_helper.processDigitArticle(answer)
        if longest_answer:
            answer = " ".join(answer.split(" ")[:longest_answer])
        return answer

    def generate_vqa(self, question_path, coco_path, year=2014, answers=None):
        questions = json.load(open(question_path))
        predictions = {}
        for i, q in tqdm(enumerate(questions['questions'][:self.args.num_samples])):
            image_path = get_coco_path(
                f'{self.args.coco_path}/{coco_path}', q['image_id'], mode='val', year=year)
            image = ImageInput(image_path)
            prompt = create_prompt(q['question'])

            few_shot_samples = []
            few_shots = random.sample(
                questions['questions'][:i]+questions['questions'][i+1:], args.few_shot)

            for sample in few_shots:
                few_shot_question_index = sample['question_id']

                if not few_shot_question_index in answers:
                    print('Not Found')
                    continue
                few_shot_image_path = get_coco_path(
                    f'{self.args.coco_path}/{coco_path}', sample['image_id'], mode='val', year=year)
                few_shot_image = ImageInput(few_shot_image_path)

                few_shot_gts = [ans['answer']
                                for ans in answers[few_shot_question_index]['answers']]
                few_shot_ans = random.choice(few_shot_gts)
                few_shot_prompt = create_prompt(
                    sample['question'], few_shot_ans)
                few_shot_samples.extend([few_shot_image, few_shot_prompt])

            prompt = few_shot_samples + [image, prompt]
            answer = self.generate(prompt)
            predictions[q['question_id']] = answer
        return predictions

    def get_vqa_accuracy(self, prediction, gt_answers):
        gtAnswers = [choice['answer'] for choice in gt_answers]
        gtAnswers = [self.clean_answer(item) for item in gtAnswers]
        longestAnswer = max([len(item.split(" ")) for item in gtAnswers])

        resAns = prediction
        resAns = self.clean_answer(resAns, longest_answer=longestAnswer)
        gtAcc = []
        acc = 0
        gtCounted = Counter(gtAnswers)
        for ans, count in gtCounted.items():
            if resAns == ans:
                acc = min(1, float(count)/3)

        return acc

    def eval_vqa_predicitions(self, predictions, answers):
        accQA = []
        for a in answers['annotations']:
            if a['question_id'] not in predictions:
                continue

            vqa_acc = self.get_vqa_accuracy(
                predictions[a['question_id']], a['answers'])

            accQA.append(vqa_acc)
        final_acc = float(sum(accQA))/len(accQA)

        return final_acc

    def generate_gqa(self, question_path, image_path):
        questions = json.load(open(question_path))
        predictions = []

        questions = list(questions.values())
        for i, val in tqdm(enumerate(questions[:self.args.num_samples])):
            image = ImageInput(os.path.join(
                image_path, f'{val["imageId"]}.jpg'))
            prompt = create_prompt(val['question'])

            few_shot_samples = []
            few_shots = random.sample(
                questions[:i]+questions[i+1:], args.few_shot)

            for sample in few_shots:

                few_shot_image = ImageInput(os.path.join(
                    image_path, f'{sample["imageId"]}.jpg'))
                few_shot_ans = sample['answer']
                few_shot_prompt = create_prompt(
                    sample['question'], few_shot_ans)
                few_shot_samples.extend([few_shot_image, few_shot_prompt])
            prompt = few_shot_samples + [image, prompt]

            answer = self.generate(prompt)
            predictions.append(
                {'answer': answer, 'gt_answer': val['answer'], 'question': prompt, 'imageId': val['imageId']})
        return predictions

    def eval_gqa_predicitions(self, predictions):
        hits = []
        for pred in predictions:
            answer = self.clean_answer(pred['answer'])
            if answer == pred['gt_answer']:
                hits.append(answer)
        final_acc = float(len(hits))/len(predictions)
        return final_acc

    def generate_vizwiz(self, question_path, image_path):
        questions = json.load(open(question_path))
        predictions = []
        for i, val in tqdm(enumerate(questions[:self.args.num_samples])):
            image = ImageInput(os.path.join(
                image_path, f'{val["image"]}'))
            prompt = create_prompt(val['question'])

            few_shot_samples = []
            few_shots = random.sample(
                questions[:i]+questions[i+1:], args.few_shot)

            for sample in few_shots:

                few_shot_image = ImageInput(os.path.join(
                    image_path, f'{sample["image"]}'))
                few_shot_gts = [ans['answer']
                                for ans in sample['answers']]
                few_shot_ans = random.choice(few_shot_gts)
                few_shot_prompt = create_prompt(
                    sample['question'], few_shot_ans)
                few_shot_samples.extend([few_shot_image, few_shot_prompt])
            prompt = few_shot_samples + [image, prompt]
            answer = self.generate(prompt)
            predictions.append(
                {'answer': answer, 'gt_answers': val['answers'], 'question': prompt, 'imageId': val['image']})
        return predictions

    def eval_vizwiz(self, predictions):
        acc_vizwiz = []
        for a in predictions:
            vqa_acc = self.get_vqa_accuracy(
                a['answer'], a['gt_answers'])

            acc_vizwiz.append(vqa_acc)
        final_acc = float(sum(acc_vizwiz))/len(acc_vizwiz)
        return final_acc


def index_vqa(answers):
    answers_dict = {}
    for a in answers['annotations']:
        answers_dict[a['question_id']] = a
    return answers_dict


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    if args.model_path is None:
        model = Magma(args.config_path).cuda()
    else:
        model = Magma.from_checkpoint(
            config_path=args.config_path, checkpoint_path=args.model_path).cuda()
    if args.aa_focused:
        model = aa_focused(model)
    datasets = args.datasets.split(',')
    experiment = Evaluator(model, args)
    rtpt = RTPT(name_initials='MM', experiment_name='Eval Script Magma',
                max_iterations=len(datasets))
    rtpt.start()
    # %%
    if 'vqa' in datasets:
        vqa_answers = json.load(
            open(f'{args.dataset_path}/VQA/v2_mscoco_val2014_annotations.json'))
        indexed_vqa = index_vqa(vqa_answers)
        vqa_predictions = experiment.generate_vqa(
            question_path=f'{args.dataset_path}/VQA/v2_OpenEnded_mscoco_val2014_questions.json', coco_path="val2014", answers=indexed_vqa)
        vqa_acc = experiment.eval_vqa_predicitions(
            vqa_predictions, vqa_answers)
        print(
            f'Accuracy for VQA of Model {args.model_path}: {vqa_acc}')
        rtpt.step()

    if 'okvqa' in datasets:
        okvqa_answers = json.load(
            open(f'{args.dataset_path}/OK_VQA/mscoco_val2014_annotations.json'))
        indexed_answers_okvqa = index_vqa(okvqa_answers)
        okvqa_predictions = experiment.generate_vqa(
            question_path=f'{args.dataset_path}/OK_VQA/OpenEnded_mscoco_val2014_questions.json', coco_path="val2014", answers=indexed_answers_okvqa)
        okvqa_acc = experiment.eval_vqa_predicitions(
            okvqa_predictions, okvqa_answers)
        print(
            f'Accuracy for OKVQA of Model {args.model_path}: {okvqa_acc}')
        rtpt.step()

    if 'gqa' in datasets:
        gqa_predictions = experiment.generate_gqa(
            question_path=f'{args.dataset_path}/gqa/testdev_all_questions.json', image_path=f'{args.dataset_path}/gqa/images')
        gqa_acc = experiment.eval_gqa_predicitions(gqa_predictions)
        print(
            f'Accuracy for GQA of Model {args.model_path}: {gqa_acc}')
        rtpt.step()

    if 'vizwiz' in datasets:
        vizwiz_predictions = experiment.generate_vizwiz(
            question_path=f'{args.dataset_path}/vizwiz/val.json', image_path=f'{args.dataset_path}/vizwiz/val')
        vizwiz_acc = experiment.eval_vizwiz(vizwiz_predictions)
        print(
            f'Accuracy for vizwiz_acc of Model {args.model_path}: {vizwiz_acc}')
        rtpt.step()
    dic = json.dumps({'vqa': vqa_acc, 'okvqa': okvqa_acc,
                      'gqa': gqa_acc, 'vizwiz': vizwiz_acc})
    with open(f"acc_{args.model_path}_shots{args.few_shot}_focused_{args.aa_focused}_samples_{args.num_samples}.json", "w") as outfile:
        outfile.write(dic)
# %%
