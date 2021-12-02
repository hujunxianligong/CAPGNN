# coding=utf-8
import os
# os.environ['TF_CUDNN_DETERMINISTIC']='1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf
import numpy as np

# import random
# seed = 100
# os.environ["PYTHONHASHSEED"] = str(seed)
# np.random.seed(seed)
# random.seed(seed)
# tf.random.set_seed(seed)

from collections import Counter

from tf_geometric.datasets.amazon_electronics import AmazonElectronicsDataset

from capgnn.layers.capgnn import CAPGNN
from tf_geometric.utils import tf_utils
import tf_geometric as tfg
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("--model_name", type=str, required=True, help="CAPGCN or CAPGAT")

parser.add_argument("--gpu_ids", type=str, required=True)

parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--l2_coef", type=float, required=True)
parser.add_argument("--cl_coef", type=float, required=True)

parser.add_argument("--input_drop_rate", type=float, required=True)
parser.add_argument("--dense_drop_rate", type=float, required=True)
parser.add_argument("--edge_drop_rate", type=float, required=True)
parser.add_argument("--coef_att_drop_rate", type=float, required=True)
parser.add_argument("--bn", default=False, action="store_true")

parser.add_argument("--num_iters", type=int, required=True)
parser.add_argument("--alpha", type=float, required=True)

parser.add_argument("--temp", type=float, required=True)
parser.add_argument("--num_views", type=int, required=True)




args = parser.parse_args()

print("args: ", args)

model_name = args.model_name
dataset = args.dataset
gpu_ids = args.gpu_ids

lr = args.lr
l2_coef = args.l2_coef
cl_coef = args.cl_coef

input_drop_rate = args.input_drop_rate
dense_drop_rate = args.dense_drop_rate
edge_drop_rate = args.edge_drop_rate
use_bn = args.bn

num_iters = args.num_iters
alpha = args.alpha

temp = args.temp
num_views = args.num_views

if model_name == "CAPGCN":
    beta = 1.0
elif model_name == "CAPGAT":
    beta = 0.3
else:
    raise Exception("model_name should be CAPGCN or CAPGAT")

coef_att_drop_rate = args.coef_att_drop_rate

num_epochs = 2000
patience = 200

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

gpu_devices = tf.config.list_physical_devices("GPU")
for gpu_device in gpu_devices:
    tf.config.set_logical_device_configuration(gpu_device, [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 6)])
    # tf.config.experimental.set_memory_growth(gpu_device, True)


if dataset in ["cora", "citeseer", "pubmed"]:
    graph, (train_index, valid_index, test_index) = tfg.datasets.PlanetoidDataset(dataset).load_data()
elif dataset in ["amazon-computers", "amazon-photo"]:
    graph = AmazonElectronicsDataset(dataset).load_data()
    num_classes = np.max(graph.y) + 1

    train_index_list = []
    valid_index_list = []
    test_index_list = []

    for label in range(num_classes):
        label_index = np.where(graph.y == label)[0]
        shuffled_label_index = np.random.permutation(label_index)
        label_train_index = shuffled_label_index[:20]
        label_valid_index = shuffled_label_index[20:50]
        label_test_index = shuffled_label_index[50:]

        train_index_list.append(label_train_index)
        valid_index_list.append(label_valid_index)
        test_index_list.append(label_test_index)

    train_index = np.concatenate(train_index_list, axis=0)
    valid_index = np.concatenate(valid_index_list, axis=0)
    test_index = np.concatenate(test_index_list, axis=0)



num_classes = graph.y.max() + 1


model = CAPGNN([64, num_classes], attention_units=1, activation=None,
               num_iterations=num_iters, alpha=alpha, beta=beta,
               input_drop_rate=input_drop_rate, dense_drop_rate=dense_drop_rate,
               edge_drop_rate=edge_drop_rate, coef_att_drop_rate=coef_att_drop_rate,
               use_bn=use_bn)


num_features = graph.x.shape[-1]


# @tf_utils.function
def forward(x, edge_index, training=False):
    return model([x, edge_index], training=training)



# @tf_utils.function
def compute_loss(logits, mask_index, vars):
    masked_logits = tf.gather(logits, mask_index)
    # tf.print("masked_logits: ", masked_logits.dtype, masked_logits)
    masked_labels = tf.gather(graph.y, mask_index)
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=masked_logits,
        labels=tf.one_hot(masked_labels, depth=num_classes)
    )

    # kernel_vals = [var for var in model.variables if "kernel" in var.name]
    kernel_vals = [var for var in vars if "kernel" in var.name]

    # for kernel_var in kernel_vals:
    #     print(kernel_var.name, tf.reduce_any(tf.math.is_nan(kernel_var)))
    l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
    cls_loss = tf.reduce_mean(losses)
    l2_loss = tf.add_n(l2_losses)

    return cls_loss + l2_loss * l2_coef, cls_loss, l2_loss


# @tf_utils.function
def evaluate(x, edge_index, current_test_index):

    with tf.GradientTape() as tape:
        logits = forward(x, edge_index, training=False)

    loss = compute_loss(logits, current_test_index, tape.watched_variables())

    masked_logits = tf.gather(logits, current_test_index)
    masked_labels = tf.gather(graph.y, current_test_index)
    y_pred = tf.argmax(masked_logits, axis=-1, output_type=tf.int32)

    corrects = tf.equal(y_pred, masked_labels)
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

    return accuracy, loss


@tf_utils.function
def evaluate_test():
    return evaluate(graph.x, graph.edge_index, test_index)


@tf_utils.function
def evaluate_val():
    return evaluate(graph.x, graph.edge_index, valid_index)


optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

val_accuracy_list = []
test_accuracy_list = []
loss_list = []
best_val_accuracy = 0
min_val_loss = 1000
final_test_accuracy = 0.0
final_step = -1
patience_counter = 0




@tf.function
def train_step():
    with tf.GradientTape() as tape:

        logits_list = []
        cls_loss_list = []
        l2_loss = None

        for i in range(num_views):
            logits = forward(graph.x, graph.edge_index, training=True)
            logits_list.append(logits)
            _, cls_loss, l2_loss = compute_loss(logits, train_index, tape.watched_variables())
            cls_loss_list.append(cls_loss)

        mean_cls_loss = tf.add_n(cls_loss_list) / num_views

        loss = mean_cls_loss + l2_loss * l2_coef

        logits_matrix = tf.stack(logits_list, axis=1)
        prob_matrix = tf.nn.softmax(logits_matrix, axis=-1)

        def compute_cl_loss(prob_matrix_a, prob_matrix_b):
            pow_prob_matrix_b = tf.math.pow(prob_matrix_b, 1.0 / temp)
            pow_prob_matrix_b /= tf.reduce_sum(pow_prob_matrix_b, axis=-1, keepdims=True)

            normed_prob_matrix_a = tf.math.l2_normalize(prob_matrix_a, axis=-1)
            normed_pow_prob_matrix_b = tf.math.l2_normalize(pow_prob_matrix_b, axis=-1)

            # stop gradient
            normed_pow_prob_matrix_b = tf.stop_gradient(normed_pow_prob_matrix_b)
            sim = normed_prob_matrix_a @ tf.transpose(normed_pow_prob_matrix_b, [0, 2, 1])
            cl_loss = -tf.reduce_mean(sim) * 2.0

            return cl_loss

        cl_loss = compute_cl_loss(prob_matrix, prob_matrix)
        loss += cl_loss * cl_coef


    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    return loss



for step in range(1, num_epochs + 1):

    loss = train_step()

    if step % 1 == 0:
        test_accuracy, _ = evaluate_test()
        val_accuracy, (_, val_loss, _) = evaluate_val()

        val_accuracy = val_accuracy.numpy()
        val_loss = val_loss.numpy()

        print("step = {}_\ttest_accuracy = {}\tval_accuracy = {}\tval_loss = {}".format(step, test_accuracy, val_accuracy, val_loss))

        if val_accuracy > best_val_accuracy or val_loss < min_val_loss:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                break

        # if val_accuracy > best_val_accuracy and val_loss < min_val_loss:
        if val_accuracy > best_val_accuracy and val_loss < min_val_loss:
            final_test_accuracy = test_accuracy
            final_step = step

            best_val_accuracy = val_accuracy
            min_val_loss = val_loss

        val_accuracy_list.append(val_accuracy)
        test_accuracy_list.append(test_accuracy)
        loss_list.append(val_loss)

        print("step = {}\tloss = {:.4f}\tval_accuracy = {:.4f}\tval_loss = {:.4f}\ttest_accuracy = {:.4f}\tfinal_test_accuracy = {:.4f}\tfinal_step = {}"
            .format(step, loss, val_accuracy, val_loss, test_accuracy, final_test_accuracy, final_step))
        print("patience_counter = {}".format(patience_counter))
        # print("cl_loss = ", test_cl_loss)
        # analysis()


print("final accuracy: {}\tfinal_step: {}".format(final_test_accuracy, final_step))
with open("results.txt", "a", encoding="utf-8") as f:
    f.write("{:.4f}\n".format(final_test_accuracy))

