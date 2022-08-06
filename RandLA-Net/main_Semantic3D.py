from os.path import join, exists
from RandLANet import Network
from tester_Semantic3D import ModelTester
from helper_ply import read_ply
from helper_tool import Plot
from helper_tool import DataProcessing as DP
from helper_tool import ConfigSemantic3D as cfg
import tensorflow as tf
import numpy as np
import pickle, argparse, os


class Semantic3D:
    def __init__(self):
        # 将数据划分为：训练集、验证集、测试集，没有label文件的是测试集，有标签文件的其中两个作为验证集，其他的为训练集
        self.name = 'Semantic3D'
        self.path = '/data/semantic3d'
        self.label_to_names = {0: 'unlabeled',
                               1: 'man-made terrain',
                               2: 'natural terrain',
                               3: 'high vegetation',
                               4: 'low vegetation',
                               5: 'buildings',
                               6: 'hard scape',
                               7: 'scanning artefacts',
                               8: 'cars'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.original_folder = join(self.path, 'original_data')
        self.full_pc_folder = join(self.path, 'original_ply')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

        # Following KPConv to do the train-validation split
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = 1

        # Initial training-validation-testing files
        self.train_files = []
        self.val_files = []
        self.test_files = []
        cloud_names = [file_name[:-4] for file_name in os.listdir(self.original_folder) if file_name[-4:] == '.txt']
        for pc_name in cloud_names:
            if exists(join(self.original_folder, pc_name + '.labels')):
                self.train_files.append(join(self.sub_pc_folder, pc_name + '.ply'))
            else:
                self.test_files.append(join(self.full_pc_folder, pc_name + '.ply'))

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)

        for i, file_path in enumerate(self.train_files):
            if self.all_splits[i] == self.val_split:
                self.val_files.append(file_path)

        self.train_files = np.sort([x for x in self.train_files if x not in self.val_files])

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Ascii files dict for testing
        # 预测数据标签名列表
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        # 调用加载数据的函数
        self.load_sub_sampled_clouds(cfg.sub_grid_size)


    # 加载采样之后的点云数据
    def load_sub_sampled_clouds(self, sub_grid_size):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = np.hstack((self.train_files, self.val_files, self.test_files))

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if file_path in self.val_files:
                cloud_split = 'validation'
            elif file_path in self.train_files:
                cloud_split = 'training'
            else:
                cloud_split = 'test'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            if cloud_split == 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

        # Get validation and test re_projection indices
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if file_path in self.val_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

            # Test projection
            if file_path in self.test_files:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        print('finished')
        return

    # Generate the input data flow
    # 数据读取生成器，按照设定的点数开始读取数据
    # 先找到概率最低的文件，然后找到该文件中概率最低的点，以这个点为中心，搜索最近的65536个点
    # 作为1次模型输入的数据
    def get_batch_gen(self, split):
        # 训练师每一轮读取的次数
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        # 验证时每一轮读取的次数
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        # 测试时每一轮读取的次数
        elif split == 'test':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        # 记录每一个点的概率值和每一个文件的最小概率值
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            # 每一个点初始化一个概率值
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            # 计算每一个文件中的所有点的最小概率值
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        if split != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

        # 抽取每一个epoch的信息
        def spatially_regular_gen():

            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                # 找出概率值最小的点云文件的序号
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                # 在该点云文件中找出概率值最小的点
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                # 提取概率最小值点的所有信息
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                # 获取概率值最小的点的信息作为中心点
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                # 给初始中心点添加噪声，生成新中心点
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                # 使用kdtree文件搜索中心点周围最近的pick_point（65536）的点的序号
                query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                # 打乱点的序号，（点云的无序性？？？？？    ）
                query_idx = DP.shuffle_idx(query_idx)

                # Get corresponding points and colors based on the index
                # 根据索引获得周围点的坐标信息
                queried_pc_xyz = points[query_idx]
                # 计算周围点相对于中心点的相对坐标
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                # 获取颜色信息
                queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]
                if split == 'test':
                    # 测试数据没有标签，设置为0
                    queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    queried_pt_weight = 1
                else:
                    # 获取周围点的label
                    queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
                    # 计算邻域点的加权值（总类别的比重）
                    queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])

                # Update the possibility of the selected points
                # 计算邻域点相对于中心点的欧氏距离值
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                """
                    使用距离和权重计算邻域点的概率值，距离中心点越近的点数越多的类别概率增加越大
                    这样可以防止一个点被多次选中，也可以防止类别多的点被多次选中，而类别少的很少被选中
                """
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                # 更新概率值
                self.possibility[split][cloud_idx][query_idx] += delta
                # 重新计算最小概率值
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                if True:
                    # np.savetxt()
                    # 迭代器iter每调用一次get_next()方法， yield返回一组数据
                    # 当调用次数超过num_per_epoch后，会触发tf.errors.OutOfRangeError
                    # 通过catch捕捉这个error之后就可以知道一轮操作已经完成
                    yield (queried_pc_xyz,                       # 坐标
                           queried_pc_colors.astype(np.float32), # 颜色
                           queried_pc_labels,                    # 标签
                           query_idx.astype(np.int32),           # 点云id
                           np.array([cloud_idx], dtype=np.int32))# 点云所在文件的id

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    # 数据预处理操作，包括：增强、每一层保留的点坐标
    def get_tf_mapping(self):
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            # 首先进行数据增强操作
            # 将坐标和颜色增强的结果合并作为特征
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                # 查询每个点周围的16(k_n)个最近邻点。记录id
                neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                # 保留1/subsampling_ratio[i]个点作为下一层的输入，每一层保留的比例分别为1/4,1/4,1/4,1/4,1/2
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                # 保留的1/4个点周围的最近邻的16个点的id
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                
                # 下采样之后的每个点在没有下采样之前的点中的最近邻点id, 这一步在解码器的上采样中使用
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neigh_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]


            """
            返回24个值：
            0~4   每一层的输入点坐标
            5~9   输入点周围的16个近邻点的id号
            10~14 输出保留点的id号
            15~19 上采样的id号
            20    输入点的特征
            21    输入点的标签
            22    输入点的序号
            23    输入点所在点云文件的序号
            """

            # RandLAnet中的inputs可以看到对应于以上的输出
            """
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
            """

            return input_list

        return tf_map

    # data augmentation
    # 数据增强，旋转、平移、缩放
    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
        # 随机旋转
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # 随机缩放
        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        # 添加随机噪声
        noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        rgb = features[:, :3]
        stacked_features = tf.concat([transformed_xyz, rgb], axis=-1)
        return stacked_features


    # 数据流初始化
    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        # 生成器
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        # 利用生成器创建dataset
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        # 设置batch_size
        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()

        # 设置map操作
        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        # 多线程处理
        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        # 创建迭代器
        # tf.data.Iterator相当于dataloader https://vimsky.com/examples/usage/python-tf.compat.v1.data.Iterator.from_structure-tf.html
        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        # 网络输入的最终最终形式（flat_inputs）
        self.flat_inputs = iter.get_next() # 每调用一次get_next(),生成一个batch_size的数据
        
        # 初始化
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    Mode = FLAGS.mode
    dataset = Semantic3D()
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)

    else:
        ##################
        # Visualize data #
        ##################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])
