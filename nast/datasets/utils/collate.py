import torch


def pack_by_pad(fields, num_obj):
    fields_shape = list(fields[0].shape)
    fields_shape = (len(fields), fields_shape[0], num_obj, *fields_shape[2:])
    new_fields = torch.zeros(fields_shape, dtype=fields[0].dtype)
    for idx, field in enumerate(fields):
        new_fields[idx, :, :field.shape[1]] = field
    return new_fields


def pack_by_concat(field):
    raise NotImplementedError()


def collate(batch_list, pack_type='pad'):
    num_objects = max(sample['hist']['boxes'].shape[1] for sample in batch_list)
    pack_fn = eval(f"pack_by_{pack_type}")
    batch = {'hist': dict(), 'futr': dict()}
    for time_key in ('hist', 'futr'):
        for field_key in ('image', 'timestamps', 'ego_poses'):
            if field_key not in batch_list[0][time_key]:
                continue
            batch[time_key][field_key] = torch.stack([sample[time_key][field_key] for sample in batch_list])
        for field_key in ("masks", "features", 'boxes'):
            if field_key not in batch_list[0][time_key]:
                continue
            batch[time_key][field_key] = pack_fn([sample[time_key][field_key] for sample in batch_list], num_objects)

    data = {
        'hist': batch['hist'],
        'futr': {k: v[:, :-1] for k, v in batch['futr'].items()}
    }
    gold = {k: v[:, 1:] for k, v in batch['futr'].items()}

    return data, gold
