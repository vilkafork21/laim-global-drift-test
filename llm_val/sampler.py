"""
Семплеры для преобразования данных в формат, пригодный для тестирования.
"""

import logging
from ast import literal_eval
from copy import deepcopy

import pandas as pd
from llm_val.utils import string_to_float


class Sampler:
    """
    Базовый семплер для хранения данных в формате train/val/test/oot.
    Алиасы oos/oot предоставлены для семантической ясности.
    """

    def __init__(self, train=None, val=None, test=None, oot=None):
        self.train = train
        self.val = val
        self.test = test
        self.oot = oot

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, var):
        self._check_var(var, "train")
        self._train = deepcopy(var)

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, var):
        self._check_var(var, "val")
        self._val = deepcopy(var)

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, var):
        self._check_var(var, "test")
        self._test = deepcopy(var)

    @property
    def oot(self):
        return self._oot

    @oot.setter
    def oot(self, var):
        self._check_var(var, "oot")
        self._oot = deepcopy(var)

    @property
    def oos(self):
        return self._train

    @oos.setter
    def oos(self, var):
        self.train = var

    def _check_var(self, var, var_name):
        if var is not None:
            if not isinstance(var, dict):
                raise AttributeError(f"{var_name} должен быть словарем")
            elif not (set(var.keys()) == {"X", "y"}):
                raise AttributeError(f"{var_name} должен содержать 2 ключа: X и y")


class AutoAsessorSampler(Sampler):
    """
    Семплер для данных асессоров.
    Поддерживает форматы: {'history', 'target'} или {'question', 'answer', 'target'}.
    """

    def __init__(self, agent_df: pd.DataFrame, real_df: pd.DataFrame):
        super().__init__()

        def process_df(df, name):
            df = df.copy()
            df["target"] = df["target"].apply(string_to_float)
            if "question" in df.columns and "answer" in df.columns:
                df = df[["question", "answer", "target"]].copy()
            elif "history" in df.columns:
                df = self._convert_history_to_qa(df, name=name)
            else:
                raise ValueError(
                    "DataFrame должен содержать либо 'history', либо 'question' и 'answer'"
                )
            return df

        agent_df = process_df(agent_df, "agent")
        real_df = process_df(real_df, "real")
        logging.info(f"Размер OOS (real): {len(real_df)}")
        logging.info(f"Размер OOT (agent): {len(agent_df)}")

        self.train = {
            "X": real_df[["question", "answer"]].reset_index(drop=True),
            "y": real_df[["target"]].reset_index(drop=True),
        }
        self.test = {
            "X": agent_df[["question", "answer"]].reset_index(drop=True),
            "y": agent_df[["target"]].reset_index(drop=True),
        }

    def _convert_history_to_qa(self, df: pd.DataFrame, name: str = "") -> pd.DataFrame:
        records = []
        skipped = 0
        for _, row in df.iterrows():
            try:
                history = literal_eval(row["history"])
            except (ValueError, SyntaxError, TypeError):
                skipped += 1
                continue
            target = row["target"]
            question = None
            answer = None
            for msg in history:
                message_type = msg.get("type", msg.get("role", None))
                if message_type in ("Пользователь", "human") and question is None:
                    question = msg.get("content", "")
                elif message_type in ("AI ассистент", "ai"):
                    answer = msg.get("content", "")
            records.append(
                {"question": question or "", "answer": answer or "", "target": target}
            )
        if skipped:
            logging.warning(f"[{name}] пропущено {skipped} строк с битым 'history'")
        return pd.DataFrame(records)
