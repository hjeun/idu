import tensorflow as tf

l2 = tf.contrib.keras.regularizers.l2

class Model(object):

    def __init__(self, rgb_features, flow_features, action_labels, relevance_labels, keep_prob):

        self.rgb_features = rgb_features
        self.flow_features = flow_features
        self.rgb_feat_dim = rgb_features.get_shape()[2].value
        self.flow_feat_dim = flow_features.get_shape()[2].value
        self.action_labels = action_labels
        self.relevance_labels = relevance_labels
        self.similarity_labels = tf.cast(tf.reduce_sum(tf.abs(action_labels - tf.expand_dims(action_labels[:, -1, :], 1)), axis=-1) - 2, tf.bool)
        self.keep_prob = keep_prob
        self.alpha = 0.3


    def build_network(self):
        with tf.variable_scope('Concatenater'):            
            # input : [batch_size, seq_length, feat_dim])
            feat_attn = tf.concat([self.rgb_features, self.flow_features], 2)
            feat_attn, embedding_last = self.embedding(feat_attn)

        with tf.variable_scope('IDU'):
            ## output : [batch_size, seq_length, hidden_size]            
            outputs = tf.keras.layers.IDU(512, return_sequences=True, dropout=self.keep_prob)(feat_attn)
            hidden_states = outputs[0]
            reset_gates = outputs[1][0]
            update_gates = outputs[1][1]

        with tf.variable_scope('Output'):
            logits = tf.reshape(hidden_states, [-1, 512])
            logits = tf.layers.dense(logits, 512, activation='relu')
            logits_last = tf.layers.dense(logits, 31, activation=None)
            logits_last = tf.reshape(logits_last, [-1, 16, 31])

        embedding_x = feat_attn
        embedding_h = outputs
        return logits_last, embedding_last, [embedding_x, embedding_h], [reset_gates, update_gates]

    def embedding(self, input):
        reshaped = tf.reshape(input, [-1, self.rgb_feat_dim + self.flow_feat_dim])
        embedding = tf.layers.dense(reshaped, 512, activation='relu')
        embedding_last = tf.layers.dense(embedding, 31, activation=None)
        embedding = tf.reshape(embedding, [-1, 16, 512])
        embedding_last = tf.reshape(embedding_last, [-1, 16, 31])
        return embedding, embedding_last  

    def testing_operation(self):
        logits_last, _, _, gates = self.build_network()
        probs = tf.nn.softmax(logits_last[:, -1, :])
        return probs, gates   


