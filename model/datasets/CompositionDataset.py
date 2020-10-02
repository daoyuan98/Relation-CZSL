from PIL import Image
import random

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
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
        # transform = transforms.Compose([
        #                 transforms.Resize(256),
        #                 transforms.CenterCrop(224),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize(mean, std)
        #             ])
    elif phase in ['val', 'test']:
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])

    return transform


class CompositionDataset(tdata.Dataset):

    def __init__(self, root, phase, embedding_dict=None, split='compositional-split', getitem_behavior=None, precompute_feat=True, **kwargs):
        self.root = root
        self.phase = phase
        self.getitem_behavior = phase if getitem_behavior is None else getitem_behavior
        self.split = split
        self.precompute_feat = precompute_feat

        self.feat_dim = None
        self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx  = {obj: idx for idx, obj in enumerate(self.objs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.attr2pairid = {}
        self.obj2pairid = {}
        for i, pair in enumerate(self.train_pairs):
            attr, obj = pair[0], pair[1]
            if attr not in self.attr2pairid.keys():
                self.attr2pairid[attr] = [i]
            else:
                self.attr2pairid[attr].append(i)
            if obj not in self.obj2pairid.keys():
                self.obj2pairid[obj] = [i]
            else:
                self.obj2pairid[obj].append(i)

        self.attr2dataid = {}
        self.obj2dataid  = {}
        for i, pair in enumerate(self.data):
            _, attr, obj = pair
            if attr not in self.attr2dataid.keys():
                self.attr2dataid[attr] = [i]
            else:
                self.attr2dataid[attr].append(i)
            if obj not in self.obj2dataid.keys():
                self.obj2dataid[obj] = [i]
            else:
                self.obj2dataid[obj].append(i)

        self.embedding_dict = embedding_dict
        self.att_emb_dict = None
        self.obj_emb_dict = None

        self.kneg = kwargs['kneg'] if 'kneg' in kwargs.keys() else None

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        # fix later -- affordance thing
        # return {object: all attrs that occur with obj}
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data + self.test_data if obj == _obj]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))

            candidates = [attr for (_, attr, obj) in self.train_data if obj == _obj]
            self.train_obj_affordance[_obj] = sorted(list(set(candidates)))
        
        if self.precompute_feat:
            self.feats = np.load(file=f'{self.root}/feat_{self.phase}.npy')

    def get_split_info(self):
        if self.split == 'compositional-split':
            data = torch.load(self.root + '/metadata.t7')
        else:
            data = torch.load(self.root + '/metadata-natural.t7')
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def sample_negative(self, attr, obj, free_sample=False, same_attr=None):

        # make sure at least one primitive concept is the same
        if free_sample:
            while True:
                new_attr, new_obj = self.train_pairs[np.random.choice(len(self.train_pairs))]
                if not (new_attr == attr and new_obj == obj):
                    break
        else:
            if same_attr is None:
                same_attr = (random.random() >= 0.5)
            if same_attr:
                new_attr = attr
                candidate_id = sorted(list(set(self.attr2pairid[new_attr]).difference(set(self.obj2pairid[obj]))))
            else:
                new_obj = obj
                candidate_id = sorted(list(set(self.obj2pairid[new_obj]).difference(set(self.attr2pairid[attr]))))
            if len(candidate_id) > 0:
                new_id = random.sample(candidate_id, 1)[0]
                new_attr, new_obj = self.train_pairs[new_id]
            else:
                # however, if that fails, fall back to free sample
                while True:
                    new_attr, new_obj = self.train_pairs[np.random.choice(len(self.train_pairs))]
                    if not (new_attr == attr and new_obj == obj):
                        break
        # select an image with category (new_attr, new_obj)
        data_candidate_id = sorted(list(set(self.attr2dataid[new_attr]).intersection(self.obj2dataid[new_obj])))
        data_id = np.random.choice(data_candidate_id, 1)[0]
        if self.precompute_feat:
            img = torch.FloatTensor(self.feats[data_id]).float()
        else:
            img_id = self.data[data_id][0]
            img = self.loader(img_id)
            img = self.transform(img)
        # return self.sample_negative(attr, obj)
        # if new_attr != attr and new_obj != obj:
        #     return self.sample_negative(attr, obj)
        # return (self.attr2idx[new_attr], self.obj2idx[new_obj])
        return img, self.att_emb_dict[new_attr], self.obj_emb_dict[new_obj], new_attr, new_obj

    def sample_negative_standalone(self, attr_idx, obj_idx):
        attr = self.attrs[attr_idx]
        obj = self.objs[obj_idx]
        _, _, neg_attr, neg_obj = self.sample_negative(attr, obj)
        return self.attr2idx[neg_attr], self.obj2idx[neg_obj], self.att_emb_dict[neg_attr], self.obj_emb_dict[neg_obj]

    def sample_positive(self, attr, obj, index=None):
        """
        Args:
            index: if set, those indices will be excluded from final sample candidates.

        Returns:
            img, att_emb, obj_emb, att, obj
        """
        data_candidate_id = sorted(list(set(self.attr2dataid[attr]).intersection(self.obj2dataid[obj]).difference(set([index]))))
        if len(data_candidate_id) > 0:
            data_id = np.random.choice(data_candidate_id, 1)[0]
        else:
            data_id = index
        if self.precompute_feat:
            img = torch.FloatTensor(self.feats[data_id]).float()
        else:
            img_id = self.data[data_id][0]
            img = self.loader(img_id)
            img = self.transform(img)
        return img, self.att_emb_dict[attr], self.obj_emb_dict[obj], attr, obj
    
    def sample_negative_by_pair_id(self, pair_id):
        data = []
        for i in range(len(self.train_pairs)):
            if i == pair_id:
                continue
            att, obj = self.train_pairs[i]
            att_id, obj_id = self.attr2idx[att], self.obj2idx[obj]
            # data_candidate_id = list(set(self.attr2dataid[att]).intersection(self.obj2dataid[obj]))
            # data_id = np.random.choice(data_candidate_id, 1)[0]
            # img_id = self.data[data_id][0]
            # img = self.loader(img_id)
            # img = self.transform(img)
            data.append([att_id, obj_id])
        return data

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        if self.precompute_feat:
            img = torch.FloatTensor(self.feats[index]).float()  # !!!
            # if self.phase == 'train':
            # img += torch.randn(img.shape)  # Normal(0, 1)
        else:
            img = self.loader(image)
            img = self.transform(img)

        # data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        data = [img, self.att_emb_dict[attr], self.obj_emb_dict[obj],
                self.attr2idx[attr], self.obj2idx[obj], attr, obj]

        if self.getitem_behavior == 'train' and self.kneg is not None:
            for k in range(self.kneg):
                img_n, neg_attr_emb, neg_obj_emb, neg_attr, neg_obj = self.sample_negative(attr, obj, same_attr=True)  # negative example for triplet loss
                data += [img_n, neg_attr_emb, neg_obj_emb, self.attr2idx[neg_attr], self.obj2idx[neg_obj], neg_attr, neg_obj]
            # for k in range(self.kneg):
            #     img_p, pos_attr_emb, pos_obj_emb, pos_attr, pos_obj = self.sample_positive(attr, obj, index)  # positive example for triplet loss
            #     data += [img_p, pos_attr_emb, pos_obj_emb, self.attr2idx[pos_attr], self.obj2idx[pos_obj], pos_attr, pos_obj]
            
                img_n_1, neg_attr_emb_1, neg_obj_emb_1, neg_attr_1, neg_obj_1 = self.sample_negative(attr, obj, same_attr=False)  # negative example for triplet loss
                data += [img_n_1, neg_attr_emb_1, neg_obj_emb_1, self.attr2idx[neg_attr_1], self.obj2idx[neg_obj_1], neg_attr_1, neg_obj_1]
            
            # img_p, attr_emb_p, obj_emb_p, attr_p, obj_p = self.sample_positive(attr, obj, index)  # negative example for triplet loss
            # data += [img_p, attr_emb_p, obj_emb_p, self.attr2idx[attr_p], self.obj2idx[obj_p]]

            # data += [np.array(self.sample_negative_by_pair_id(self.train_pairs.index((attr, obj))))]
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
