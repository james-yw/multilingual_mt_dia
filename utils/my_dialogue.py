from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    Dictionary
)

from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    num_buckets=0,
    shuffle=True,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        src_dataset = PrependTokenDataset(src_dataset, src_dict.index("<dia>"))
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
    )

@register_task('my_dialogue')
class MyDialogueTask(TranslationTask):
    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename and add the special token <mt> <dia>

        Args:
            filename (str): the filename
        """
        dicts = Dictionary.load(filename)
        dicts.add_symbol("<mt>")
        dicts.add_symbol("<dia>")
        dicts.add_symbol("__en__")
        dicts.add_symbol("__zh__")
        dicts.add_symbol("__de__")
        dicts.add_symbol("__fr__")

        return dicts

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != 'test'),
        )
       
        print("##################",split)
        print(self.datasets[split][0])
        
        print(self.src_dict.string(self.datasets[split][0]['source']))
        print(self.src_dict.string(self.datasets[split][0]['target']))
