# coding=utf-8
import tensorflow as tf
from tf_geometric import SparseAdj
from tf_geometric.nn.kernel.segment import segment_softmax
from tf_geometric.utils.graph_utils import add_self_loop_edge


class CAPGNN(tf.keras.Model):

    def __init__(self, units_list,
                 attention_units=None,
                 activation=None,
                 num_iterations=10,
                 alpha=0.1,
                 beta=0.3,  # CAPGCN: beta=1.0    CAPGAT: beta=0.3
                 dense_activation=tf.nn.relu,
                 query_activation=tf.nn.relu,
                 key_activation=tf.nn.relu,
                 input_drop_rate=0.0,
                 dense_drop_rate=0.0,
                 edge_drop_rate=0.0,
                 coef_att_drop_rate=0.3,
                 use_bn=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.units_list = units_list
        self.attention_units = units_list[-1] if attention_units is None else attention_units

        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta

        self.query_kernel = None
        self.query_bias = None
        self.query_activation = query_activation

        self.key_kernel = None
        self.key_bias = None
        self.key_activation = key_activation

        self.activation = activation

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.dense_drop_rate = dense_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.coef_att_drop_rate = coef_att_drop_rate

        self.use_bn = use_bn
        if use_bn:
            self.input_bn = tf.keras.layers.BatchNormalization()

        self.input_dropout = tf.keras.layers.Dropout(input_drop_rate)

        self.value_dense = tf.keras.Sequential()

        for i, units in enumerate(units_list):
            self.value_dense.add(tf.keras.layers.Dense(units))

            if i < len(units_list) - 1:
                if use_bn:
                    self.value_dense.add(tf.keras.layers.BatchNormalization())
                self.value_dense.add(tf.keras.layers.Activation(dense_activation))
                self.value_dense.add(tf.keras.layers.Dropout(dense_drop_rate))

        self.coef_att_dropout = tf.keras.layers.Dropout(coef_att_drop_rate)
        self.attention_logits = self.add_weight("attention_logits_kernel", shape=[num_iterations], initializer="zeros")


    def build(self, input_shapes):
        x_shape = input_shapes[0]
        num_features = x_shape[-1]

        self.query_kernel = self.add_weight("query_kernel", shape=[num_features, self.attention_units],
                                            initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.query_bias = self.add_weight("query_bias", shape=[self.attention_units],
                                          initializer="zeros", regularizer=self.bias_regularizer)

        self.key_kernel = self.add_weight("key_kernel", shape=[num_features, self.attention_units],
                                          initializer="glorot_uniform", regularizer=self.kernel_regularizer)
        self.key_bias = self.add_weight("key_bias", shape=[self.attention_units],
                                        initializer="zeros", regularizer=self.bias_regularizer)

    def call(self, inputs, training=None, mask=None, num_iterations=None):
        """

        :param inputs: List of graph info: [x, edge_index] or [x, edge_index, edge_weight].
            Note that the edge_weight will not be used.
        :return: Updated node features (x), shape: [num_nodes, units]
        """
        x, edge_index = inputs[0], inputs[1]

        if self.use_bn:
            x = self.input_bn(x, training=training)

        x = self.input_dropout(x, training=training)

        num_nodes = tf.shape(x)[0]

        # self-attention
        edge_index, _ = add_self_loop_edge(edge_index, num_nodes)
        deg = tf.math.unsorted_segment_sum(tf.ones_like(edge_index[0], dtype=tf.float32), edge_index[0], num_nodes)
        row, col = edge_index[0], edge_index[1]
        V = self.value_dense(x, training=training)

        # Transformer-based GAT
        def attention(query_kernel, query_bias, query_activation, key_kernel, key_bias, key_activation):
            Q = x @ query_kernel + query_bias
            if query_activation is not None:
                Q = query_activation(Q)
            Q = tf.gather(Q, row)

            K = x @ key_kernel + key_bias
            if key_activation is not None:
                K = key_activation(K)
            K = tf.gather(K, col)

            scale = tf.math.sqrt(tf.cast(tf.shape(Q)[-1], tf.float32))
            # att_score_ = tf.reduce_sum(Q_ * K_, axis=-1) / scale
            att_score = tf.reduce_sum(Q * K, axis=-1) / scale

            normed_att_score = segment_softmax(att_score, edge_index[0], num_nodes)
            return normed_att_score

        gat_affinity = attention(self.query_kernel, self.query_bias, self.query_activation, self.key_kernel,
                                      self.key_bias, self.key_activation)


        deg_sqrt = tf.pow(deg, 0.5)
        deg_inv_sqrt = tf.pow(deg, -0.5)
        deg_inv_sqrt = tf.where(
            tf.math.logical_or(tf.math.is_inf(deg_inv_sqrt), tf.math.is_nan(deg_inv_sqrt)),
            tf.zeros_like(deg_inv_sqrt),
            deg_inv_sqrt
        )

        renormed_gat_affinity = tf.gather(deg_sqrt, row) * gat_affinity * tf.gather(deg_inv_sqrt, col)

        gcn_affinity = tf.gather(deg_inv_sqrt, row) * tf.ones_like(edge_index[0], dtype=tf.float32) * tf.gather(deg_inv_sqrt, col)
        affinity = gcn_affinity * self.beta + (1.0 - self.beta) * renormed_gat_affinity

        h = V

        def propagate(h):

            h_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            for _ in range(self.num_iterations):

                if training and self.edge_drop_rate > 0.0:
                    current_affinity = tf.compat.v2.nn.dropout(affinity, self.edge_drop_rate)
                else:
                    current_affinity = affinity

                current_adj = SparseAdj(edge_index, current_affinity, [num_nodes, num_nodes])
                h = current_adj @ h

                h = h * (1.0 - self.alpha) + V * self.alpha
                h_list = h_list.write(h_list.size(), h)

            h_matrix = h_list.stack()

            return h_matrix

        h_matrix = propagate(h)

        hop_attention_score = tf.nn.softmax(tf.nn.leaky_relu(self.attention_logits, alpha=0.2), axis=-1)
        hop_attention_score = tf.expand_dims(tf.expand_dims(hop_attention_score, axis=-1), axis=-1)
        hop_attention_score = tf.tile(hop_attention_score, [1, tf.shape(h_matrix)[1], tf.shape(h_matrix)[2]])
        hop_attention_score = self.coef_att_dropout(hop_attention_score, training=training)
        hop_attention_score /= (tf.reduce_sum(hop_attention_score, axis=0, keepdims=True) + 1e-8)
        weighted_h_matrix = h_matrix * hop_attention_score
        h = tf.reduce_sum(weighted_h_matrix, axis=0)

        if self.activation is not None:
            h = self.activation(h)

        return h



