from PIL import Image

import numpy as np
import torch
import torch.utils.data as tdata
import torchvision.transforms as transforms


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Phase can only be \"train\" or \"test\".')

    return transform


class CompositionDataset(tdata.Dataset):

    def __init__(self, root, phase, embedding_dict=None, split='compositional-split'):
        self.root = root
        self.phase = phase
        self.split = split

        self.feat_dim = None
        self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, self.train_pairs, self.test_pairs = self.parse_split()
        assert len(set(self.train_pairs) & set(
            self.test_pairs)) == 0, 'train and test are not mutually exclusive'
        self.train_data, self.test_data = self.get_split_info()
        self.data = self.train_data if self.phase == 'train' else self.test_data

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        self.embedding_dict = embedding_dict
        self.att_emb_dict = None
        self.obj_emb_dict = None

        print(
            '# train pairs: %d | # test pairs: %d' % (len(self.train_pairs), len(self.test_pairs)))

        # fix later -- affordance thing
        # return {object: all attrs that occur with obj}
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data + self.test_data if
                          obj == _obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj == _obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

    def get_split_info(self):

        data = torch.load(self.root + '/metadata.t7')
        train_pair_set = set(self.train_pairs)
        train_data, test_data = [], []
        for instance in data:

            image, attr, obj = instance['image'], instance['attr'], instance['obj']

            if attr == 'NA' or (attr, obj) not in self.pairs:
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if (attr, obj) in train_pair_set:
                train_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, test_data

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs('%s/%s/train_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(list(set(tr_attrs + ts_attrs))), sorted(
            list(set(tr_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, ts_pairs

    def sample_negative(self, attr, obj):
        new_attr, new_obj = self.train_pairs[np.random.choice(len(self.train_pairs))]
        if new_attr == attr and new_obj == obj:
            return self.sample_negative(attr, obj)
        # return (self.attr2idx[new_attr], self.obj2idx[new_obj])
        return self.att_emb_dict[new_attr], self.obj_emb_dict[new_obj], new_attr, new_obj

    def sample_negative_standalone(self, attr_idx, obj_idx):
        attr = self.attrs[attr_idx]
        obj = self.objs[obj_idx]
        _, _, neg_attr, neg_obj = self.sample_negative(attr, obj)
        return self.attr2idx[neg_attr], self.obj2idx[neg_obj], self.att_emb_dict[neg_attr], self.obj_emb_dict[neg_obj]

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        # data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        data = [img, self.att_emb_dict[attr], self.obj_emb_dict[obj],
                self.attr2idx[attr], self.obj2idx[obj], attr, obj]

        if self.phase == 'train':
            neg_attr_emb, neg_obj_emb, neg_attr, neg_obj = self.sample_negative(attr, obj)  # negative example for triplet loss
            data += [neg_attr_emb, neg_obj_emb, self.attr2idx[neg_attr], self.obj2idx[neg_obj]]
        return data

    def __len__(self):
        return len(self.data)


# for debug only
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from model.datasets.glove import load_glove_as_dict

    train_dataset = CompositionDataset('../../data/mitstates', 'train',
                                       embedding_dict=load_glove_as_dict('../../data/glove'))
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    data = next(iter(train_dataloader))
    pass
