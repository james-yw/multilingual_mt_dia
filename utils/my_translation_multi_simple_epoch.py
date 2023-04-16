import os, torch

import logging
import datetime
import time

from fairseq.data import (
    data_utils,
    FairseqDataset,
    iterators,
    LanguagePairDataset,
    ListDataset,
    PrependTokenDataset,
    Dictionary
)

from fairseq.tasks import register_task
from fairseq.data.multilingual.sampling_method import SamplingMethod


from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager



###
def get_time_gap(s, e):
    return (datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)).__str__()
###


logger = logging.getLogger(__name__)


@register_task('my_translation_multi_simple_epoch')
class MyTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):
    @staticmethod
    def add_args(parser):
        TranslationMultiSimpleEpochTask.add_args(parser)
        pass

   


    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename and add the special token <mt> <dia>

        Args:
            filename (str): the filename
        """
        dicts = Dictionary.load(filename)
        dicts.add_symbol("<mt>")
        dicts.add_symbol("<dia>")
        return dicts


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split in self.datasets:
            dataset = self.datasets[split]
            if self.has_sharded_data(split) and dataset.load_next_shard:
                shard_epoch = dataset.shard_epoch
            else:
                # no need to load next shard so skip loading
                # also this avoid always loading from beginning of the data
                return
        else:
            shard_epoch = None
        logger.info(f'loading data for {split} epoch={epoch}/{shard_epoch}')
        self.datasets[split] = self.data_manager.load_sampled_multi_epoch_dataset(
            split,
            self.training,
            epoch=epoch, combine=combine, shard_epoch=shard_epoch, **kwargs
        )
    
        print("##################",split)
        print(self.dicts)
        print(self.datasets[split][0])
        if split=='train':
            print(self.dicts['en'].string(self.datasets[split][0][1]['source']))
            print(self.dicts['en'].string(self.datasets[split][0][1]['target']))
        else:
            print(self.dicts['en'].string(self.datasets[split][0]['source']))
            print(self.dicts['en'].string(self.datasets[split][0]['target']))

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary)
        src_langtok_spec, tgt_langtok_spec = self.args.langtoks['main']
        print("%%%%%%%%%%%%%%%%%%")
        print(src_langtok_spec)
        print(tgt_langtok_spec)
        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                    dataset,
                    src_eos=self.source_dictionary.eos(),
                    src_lang=self.args.source_lang,
                    tgt_eos=self.target_dictionary.eos(),
                    tgt_lang=self.args.target_lang,
                    src_langtok_spec=src_langtok_spec,
                    tgt_langtok_spec=tgt_langtok_spec,
                )
        else:
            dataset.src = self.data_manager.src_dataset_tranform_func(
                self.args.source_lang,
                self.args.target_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
                )
            print("************************")   
            print(dataset.src[0])
        return dataset