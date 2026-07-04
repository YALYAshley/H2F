import csv
import json
import os
import string

from torch.utils.data import Dataset

from lavis.datasets.data_utils import load_video


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("data", "annotations", "questions"):
        if isinstance(data.get(key), list):
            return data[key]
    # MVBench is also distributed as a mapping from task name to samples.
    samples = []
    for task, task_samples in data.items():
        if isinstance(task_samples, list):
            for sample in task_samples:
                sample = dict(sample)
                sample.setdefault("task_type", task)
                samples.append(sample)
    return samples


def _resolve_answer(answer, choices):
    if isinstance(answer, int):
        return choices[answer]
    answer = str(answer).strip()
    if answer.isdigit() and int(answer) < len(choices):
        return choices[int(answer)]
    if len(answer) == 1 and answer.upper() in string.ascii_uppercase:
        index = string.ascii_uppercase.index(answer.upper())
        if index < len(choices):
            return choices[index]
    if answer.startswith("(") and len(answer) > 2:
        index = string.ascii_lowercase.find(answer[1].lower())
        if 0 <= index < len(choices):
            return choices[index]
    return answer


class MultipleChoiceVideoQADataset(Dataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_paths,
        num_frames,
        prompt="",
        split="train",
        dataset_name="",
    ):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vis_root = vis_root
        self.num_frames = num_frames
        self.prompt = prompt
        self.split = split
        self.dataset_name = dataset_name
        raw_annotations = []
        for ann_path in ann_paths:
            if ann_path.lower().endswith(".csv"):
                with open(ann_path, "r", encoding="utf-8-sig") as f:
                    raw_annotations.extend(list(csv.DictReader(f)))
            else:
                raw_annotations.extend(_load_json(ann_path))

        self.annotation = {}
        for index, sample in enumerate(raw_annotations):
            sample = dict(sample)
            question_id = str(
                self._first(sample, ("question_id", "qid", "id"), index)
            )
            choices = self._parse_choices(sample)
            raw_answer = self._first(
                sample, ("answer", "answer_idx", "correct", "label")
            )
            sample["answer"] = self.text_processor(
                _resolve_answer(raw_answer, choices)
            )
            sample["question_id"] = question_id
            self.annotation[question_id] = sample
        self.question_ids = list(self.annotation.keys())

    @staticmethod
    def _first(sample, names, default=None):
        for name in names:
            value = sample.get(name)
            if value is not None and value != "":
                return value
        return default

    def _parse_choices(self, sample):
        choices = self._first(
            sample, ("candidates", "choices", "options", "answer_choices")
        )
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except json.JSONDecodeError:
                choices = [x.strip() for x in choices.split("|")]
        if not choices:
            choices = [
                sample[key]
                for key in ("q0", "q1", "q2", "q3", "q4")
                if sample.get(key) not in (None, "")
            ]
        return [str(choice) for choice in choices]

    def _video_path(self, sample):
        video = str(
            self._first(
                sample,
                ("video", "video_path", "video_name", "video_id", "vid"),
            )
        )
        if not os.path.splitext(video)[1]:
            extension = ".mp4"
            video = video + extension
        return video if os.path.isabs(video) else os.path.join(self.vis_root, video)

    def __getitem__(self, index):
        question_id = self.question_ids[index]
        sample = self.annotation[question_id]
        choices = self._parse_choices(sample)
        question = str(self._first(sample, ("question", "query", "Q")))
        options = " ".join(
            "({}) {}".format(string.ascii_lowercase[i], choice)
            for i, choice in enumerate(choices)
        )
        question = "{} Options: {}".format(question, options)
        question = self.text_processor(question)
        if self.prompt:
            question = self.prompt.format(question)

        answer = sample["answer"]
        video = load_video(
            self._video_path(sample),
            n_frms=self.num_frames,
            sampling="uniform",
        )
        video = self.vis_processor(video)
        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": question_id,
        }

    def __len__(self):
        return len(self.question_ids)


class MVBenchDataset(MultipleChoiceVideoQADataset):
    def __init__(self, *args, **kwargs):
        kwargs["dataset_name"] = "mvbench"
        super().__init__(*args, **kwargs)


class NExTQADataset(MultipleChoiceVideoQADataset):
    def __init__(self, *args, **kwargs):
        kwargs["dataset_name"] = "nextqa"
        super().__init__(*args, **kwargs)
