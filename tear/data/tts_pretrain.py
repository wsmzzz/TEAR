import lmdb
import torch

from ..data.tts_data_pb2 import Datum
from sklearn.preprocessing import StandardScaler
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from ..data.utils import *
datum = Datum()

logger = logging.getLogger(__name__)


# @dataclasses
# class TTSPretrainDatasetItem(object):
#     index:int
#     index:int


class TTSPretrainDataset(RawAudioDataset):
    def __init__(
            self,
            datadir,
            sample_rate=16000,
            max_frame=None,
            min_frame=0,
            shuffle=True,
            pad=True,
            normalize=False,
            bucket_size=128,
            compute_mask_indices=False,
            **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        """
        lmdb index  speaker name       filename    frame_number
        0000000000 chengjingjing_2.19h 00001039
        0000000001 chengjingjing_2.19h 00000171
        """
        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        with open(datadir, 'r') as reader:
            lines = reader.readlines()
            self.rootdir = lines[0].strip()
            self.statsdir = lines[1].strip()
            for line in lines[2:]:
                line = line.strip()
                lmdb_index, speaker, filename, sz = line.split(' ')
                sz = int(sz)
                if sz < min_frame or (max_frame is not None and sz > max_frame):
                    skipped += 1
                    self.skipped_indices.add(lmdb_index)
                    continue
                self.fnames.append(line)
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")
        self.sizes = np.array(sizes, dtype=np.int64)

        mel_mean =  read_hdf5(self.statsdir,'mel_mean')
        mel_std = read_hdf5(self.statsdir,'mel_std')
        self.mel_scaler = StandardScaler()
        self.mel_scaler.mean_ = mel_mean
        self.mel_scaler.scale_ = mel_std

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    def __getitem__(self, index):
        # logging.info(self.rootdir)
        env = lmdb.open(self.rootdir,readonly=True,lock=False)

        def gen_g2p_align(grapheme_lens):
            len_phoneme = np.sum(grapheme_lens)

            def gen(input):
                x = input[0]
                y = input[1]
                pre = np.sum(grapheme_lens[0:x]).astype(np.int)
                return [0] * pre + [1] * y + [0] * (len_phoneme - pre - y)

            return np.asarray(list(map(gen, enumerate(grapheme_lens))))



        with env.begin(write=False) as txn:
            datum.ParseFromString(txn.get(self.fnames[index].encode()))
            file_name = self.fnames[index].strip().split(' ')[2]
            phoneme_id = np.frombuffer(datum.phoneme_id, dtype=np.int32)
            dur = np.frombuffer(datum.duration, dtype=np.int32)
            pitch = np.frombuffer(datum.pitch, dtype=np.int32)
            energy = np.frombuffer(datum.energy, dtype=np.float32)
            src_tokens = np.frombuffer(datum.src_tokens, dtype=np.int32)
            total_frame = np.sum(dur)
            mel = np.frombuffer(datum.mel, dtype=np.float32).reshape([-1, 80])
            mel = self.mel_scaler.transform(mel)
            wav = np.frombuffer(datum.wav, dtype=np.float32).reshape([-1])
            align,start_mat,end_mat,start_mark,end_mark  = dur_to_align(dur)
            l_mask = longformer_mask(dur)
            #remove silence
            grapheme_lens = np.frombuffer(datum.char_lens, dtype=np.int32)[1:-1]
            grapheme_pos = []

            grapheme_padding = np.arange(1,1+len(grapheme_lens))
            for i ,j in enumerate(grapheme_lens):
                grapheme_pos += [i+1]*j
            grapheme_pos = np.asarray(grapheme_pos)
            assert grapheme_pos.shape[0]==phoneme_id.shape[0]
            g2p_align = gen_g2p_align(grapheme_lens)






        return {"index": index,
                "phoneme_id": phoneme_id,
                'src_tokens': src_tokens,
                "dur": dur,
                "pitch": pitch,
                "energy": energy,
                "mel": mel,
                "wav": wav,
                'grapheme_pos':grapheme_pos,
                "grapheme_padding":grapheme_padding,
                 "g2p_align":g2p_align,
                'mel_len': mel.shape[0],
                'phone_len': phoneme_id.shape[0],
                'align': align,
                'start_mat':start_mat,
                'end_mat':end_mat,
                'start_mark':start_mark,
                'grapheme_lens':grapheme_lens,
                'end_mark':end_mark,
                'l_mask':l_mask,
                'file_name':file_name,
                }

    def collater(self, batch):


        if len(batch) == 0:
            return {}

        index = [data['index'] for data in batch]
        phoneme_id = [data['phoneme_id'] for data in batch]
        duration = [data['dur'] for data in batch]
        pitch = [data['pitch'] for data in batch]
        energy = [data['energy'] for data in batch]
        file_name = [data['file_name'] for data in batch]
        mel = [data['mel'] for data in batch]
        wav = [data['wav'] for data in batch]
        mel_len = [data['mel_len'] for data in batch]
        phone_len = [data['phone_len'] for data in batch]
        align = [data['align'] for data in batch]
        start_mat = [data['start_mat'] for data in batch]
        end_mat = [data['end_mat'] for data in batch]
        start_mark = [data['start_mark'] for data in batch]
        end_mark = [data['end_mark'] for data in batch]
        l_mask = [data['l_mask'] for data in batch]
        grapheme_pos = [data['grapheme_pos'] for data in batch]
        g2p_align = [data['g2p_align'] for data in batch]
        grapheme_padding =  [data['grapheme_padding'] for data in batch]
        max_mel = max(mel_len)
        max_phone = max(phone_len)
        max_grapheme = max([len(i) for i in grapheme_padding])





        seg_pos =[]
        sil_pos = []

        for ids  in phoneme_id:
            seg, sil = seq_pos_encoding(ids, sil_id=[0,265,257,258,259])
            seg +=1
            sil +=1
            seg_pos.append(seg)
            sil_pos.append(sil)



        phoneme_id = torch.LongTensor(pad_1d(phoneme_id, max_len=max_phone))
        duration = torch.LongTensor(pad_1d(duration, max_len=max_phone))
        mel_feats = torch.FloatTensor(pad_2d(mel, max_len=max_mel))
        align = torch.FloatTensor(pad_weight(align,max_x_len=max_phone,max_y_len=max_mel))
        start_mat = torch.FloatTensor(pad_weight(start_mat,max_x_len=max_phone,max_y_len=max_mel))
        end_mat = torch.FloatTensor(pad_weight(end_mat,max_x_len=max_phone,max_y_len=max_mel))
        g2p_align = torch.FloatTensor(pad_weight(g2p_align,max_x_len=max_grapheme,max_y_len=max_phone))
        l_mask = torch.LongTensor(pad_2d(l_mask,max_len=max_mel))
        start_mark = torch.LongTensor(pad_1d(start_mark,max_len=max_mel))
        end_mark =  torch.LongTensor(pad_1d(end_mark,max_len=max_mel))
        pitch = torch.FloatTensor(pad_1d(pitch,max_len=max_phone))
        seg_pos =  torch.LongTensor(pad_1d(seg_pos,max_len=max_phone))
        sil_pos = torch.LongTensor(pad_1d(sil_pos,max_len=max_phone))
        mel_pos = torch.LongTensor(sequence_mask(mel_len,max_mel))
        grapheme_pos =  torch.LongTensor(pad_1d(grapheme_pos,max_len=max_phone))
        grapheme_mask = torch.LongTensor(pad_1d(grapheme_padding,max_len=max_grapheme))



        out = {'index':index}
        out['id'] = torch.LongTensor(index)
        out['length'] = phone_len
        out['file_name'] =file_name

        out["net_input"] ={
                           'duration':duration,
                           'mel_feats':mel_feats,
                           'align':align,
                           'pitch':pitch,
                           'seg_pos':seg_pos,
                           'sil_pos':sil_pos,
                           'mel_pos':mel_pos,
                           'grapheme_pos':grapheme_pos,
                            "grapheme_mask":grapheme_mask,
                           'start_mat': start_mat,
                           'end_mat': end_mat,
                           'start_mark': start_mark,
                           'end_mark': end_mark,
                           'l_mask': l_mask,
                            "g2p_align":g2p_align,

                           }


        return out




    def ordered_indices(self):

        if self.shuffle:
            import random
            index_sort = np.argsort(self.sizes).tolist()
            final_index = []
            iter = len(self) // self.bucket_size
            for i in range(iter):
                sub_list = index_sort[i * self.bucket_size:(i + 1) * self.bucket_size]
                random.shuffle(sub_list)
                final_index += sub_list
            final_index = final_index + index_sort[iter*self.bucket_size:]
            return np.asarray(final_index, dtype=np.int64)
        else:
            return np.arange(len(self))



class FrameDataset(RawAudioDataset):
    def __init__(
            self,
            datadir,
            sample_rate=16000,
            max_frame=None,
            min_frame=0,
            shuffle=True,
            pad=True,
            normalize=False,
            bucket_size=128,
            compute_mask_indices=False,
            **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        """
        lmdb index  speaker name       filename    frame_number
        0000000000 chengjingjing_2.19h 00001039
        0000000001 chengjingjing_2.19h 00000171
        """
        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        with open(datadir, 'r') as reader:
            lines = reader.readlines()
            self.rootdir = lines[0].strip()
            self.statsdir = lines[1].strip()
            for line in lines[2:]:
                line = line.strip()
                lmdb_index, speaker, filename, sz = line.split(' ')
                sz = int(sz)
                if sz < min_frame or (max_frame is not None and sz > max_frame):
                    skipped += 1
                    self.skipped_indices.add(lmdb_index)
                    continue
                self.fnames.append(line)
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")
        self.sizes = np.array(sizes, dtype=np.int64)

        mel_mean =  read_hdf5(self.statsdir,'mel_mean')
        mel_std = read_hdf5(self.statsdir,'mel_std')
        self.mel_scaler = StandardScaler()
        self.mel_scaler.mean_ = mel_mean
        self.mel_scaler.scale_ = mel_std

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    def __getitem__(self, index):
        # logging.info(self.rootdir)
        env = lmdb.open(self.rootdir,readonly=True,lock=False)


        with env.begin(write=False) as txn:

            wav = np.frombuffer(datum.wav, dtype=np.float32).reshape([-1])
            logger.info(wav)
            wav_len =  len(wav)

            padding_mask = np.arange(1,1+wav_len)




        return {"index": index,
                "wav": wav,
                "wav_len":wav_len,
                "padding_mask":padding_mask
                }

    def collater(self, batch):

        if len(batch) == 0:
            return {}

        index = [data['index'] for data in batch]

        wav = [data['wav'] for data in batch]
        wav_len = [data['wav_len'] for data in batch]
        padding_mask = [data['padding_mask'] for data in batch]
        max_wav= max(wav_len)
        wav = torch.LongTensor(pad_1d(wav,max_len=max_wav,pad_val=-10))
        padding_mask =torch.LongTensor(pad_1d(padding_mask,max_len=max_wav,pad_val=-10))
        out = {'index':index}
        out["net_input"] ={
                           'source':wav,
                            "padding_mask":padding_mask,
                           }


        return out

    def ordered_indices(self):

        if self.shuffle:
            import random
            index_sort = np.argsort(self.sizes).tolist()
            final_index = []
            iter = len(self) // self.bucket_size
            for i in range(iter):
                sub_list = index_sort[i * self.bucket_size:(i + 1) * self.bucket_size]
                random.shuffle(sub_list)
                final_index += sub_list
            final_index = final_index + index_sort[iter*self.bucket_size:]
            return np.asarray(final_index, dtype=np.int64)
        else:
            return np.arange(len(self))
if __name__ == "__main__":
    data = TTSPretrainDataset(datadir='/yrfs1/intern/smwang9/tts_pretrain_data/tts_data/train.key',
                              sample_rate=16000)


    indices = data.ordered_indices()

        # filter examples that are too large

    print(data.num_tokens(100))
    print(data.num_tokens(200))
    print(data.num_tokens(300))

    batch_sampler = data.batch_by_size(
        indices,
        max_tokens=20000,
        max_sentences=None,
        required_batch_size_multiple=1,
    )
    from fairseq.data import  iterators

    epoch_iter = iterators.EpochBatchIterator(
        dataset=data,
        collate_fn=data.collater,
        batch_sampler=batch_sampler)
    iter = epoch_iter.next_epoch_itr()


    for i ,j in enumerate(iter):
        print(i)
        



