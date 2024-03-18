def get_default_config():
    drop = 0.1
    input_drop = 0.5        # unified model set to 0.5, others set to 0
    task = 'Pain'       # 'Pain' or 'KLG'
    debug = False

    return dict(
        ex_name='Pred'+task+'Debug' if debug else 'Pred'+task,
        result_folder='./result',

        data=dict(
            data_sheet_folder='./data_sheet_folder',
            label_sheet_name='WOMAC.csv' if task == 'Pain' else 'KLG_score.csv',
            tabular_sheet_name='Enrollees.csv',
            image_folder='./image_folder',
            img_size=128,
            flip=True,
            num_load=50 if debug else -1,
        ),

        training=dict(
            lr_cls=1e-5,
            lr_tab=1e-6,
            lr_xray=1e-5 if task == 'Pain' else 1e-4,
            lr_thickness=1e-6 if task == 'Pain' else 1e-5,
            batch_size=8 if debug else 256,
            epoch=30,
            augmentation=True,
            grad_clip=1.,
        ),

        img_encoder=dict(
            arch='resnet18',
        ),

        tab_encoder=dict(
            dim=32,
            n_layers=6,
            n_heads=8,
            head_dim=16,
            dropout=drop,
        ),

        attention=dict(
            input_dim=512,
            output_dim=2 if task == 'Pain' else 4,
            block_size=6,
            n_layers=6,
            n_heads=8,
            input_drop=input_drop,
            dropout=drop,
            bias=False,
        ),
    )


