from .preprocessor import Preprocessor
import os
import numpy as np
import pandas as pd
from tadr.utils.logging import get_logger
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder, QuantileTransformer


logger = get_logger(__name__)

class EDGEIOTPreprocessor(Preprocessor):
    def __init__(self):
        self.columns_to_drop = [ # done
            "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", 
            "http.file_data","http.request.full_uri","icmp.transmit_timestamp",
            "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",
            "tcp.dstport", "udp.port", "mqtt.msg", "Attack_label"
        ]
        self.features = [ #done
            'arp.opcode', 'arp.hw.size', 'icmp.checksum', 'icmp.seq_le',
            'icmp.unused', 'http.content_length', 'http.request.method',
            'http.referer', 'http.request.version', 'http.response',
            'http.tls_port', 'tcp.ack', 'tcp.ack_raw', 'tcp.checksum',
            'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn',
            'tcp.connection.synack', 'tcp.flags', 'tcp.flags.ack', 'tcp.len',
            'tcp.seq', 'udp.stream', 'udp.time_delta', 'dns.qry.name',
            'dns.qry.name.len', 'dns.qry.qu', 'dns.qry.type', 'dns.retransmission',
            'dns.retransmit_request', 'dns.retransmit_request_in',
            'mqtt.conack.flags', 'mqtt.conflag.cleansess', 'mqtt.conflags',
            'mqtt.hdrflags', 'mqtt.len', 'mqtt.msg_decoded_as', 'mqtt.msgtype',
            'mqtt.proto_len', 'mqtt.protoname', 'mqtt.topic', 'mqtt.topic_len',
            'mqtt.ver', 'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id']
        self.label_column = 'Attack_type' # Done
        self.benign_label = 'Normal' # Done
        self.encoded_categorical_features = [ # Done
            'http.request.method', 'http.referer', 'http.request.version',
            'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic', 'dns.qry.name.len'
        ]
        self.cat_features = self.encoded_categorical_features
        self.cat_features_in_large_scale = []
        self.cont_features = [
            'arp.opcode', 'arp.hw.size', 'icmp.checksum', 'icmp.seq_le', 'icmp.unused',
            'http.content_length', 'http.tls_port', 'tcp.ack', 'tcp.ack_raw', 'tcp.checksum',
            'tcp.flags', 'tcp.len','tcp.seq', 'udp.stream', 'udp.time_delta', 'dns.qry.name',
            'dns.qry.qu', 'dns.qry.type', 'dns.retransmission',
            'mqtt.conflags',
            'mqtt.hdrflags', 'mqtt.len', 'mqtt.msg_decoded_as', 'mqtt.msgtype',
            'mqtt.proto_len', 'mqtt.topic_len', 'mqtt.ver', 'mbtcp.len',
            'mbtcp.trans_id', 'mbtcp.unit_id'
        ]
        self.encoded_numerical_features = self.cat_features_in_large_scale + self.cont_features
        self.binary_features = [
            'tcp.flags.ack', 'tcp.connection.rst', 'http.response', 
            'dns.retransmit_request_in', 'tcp.connection.fin', 'dns.retransmit_request', 
            'tcp.connection.syn','tcp.connection.synack' ,'mqtt.conflag.cleansess'
        ]
        self.encoders = {}
        self.encoders_dir = None
        self.ordered_features = self.cat_features + self.binary_features + self.cont_features

    
    def select_features_and_label(self, df: pd.DataFrame):
        return df[self.features + [self.label_column]]

    def remove_missing_and_inf_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values and non-numeric strings in numeric columns with NaN, then drop."""
        result_df = df.replace([np.inf, -np.inf], np.nan)
        # Coerce mixed-type numeric columns (e.g. 'dns.qry.name.len', 'mqtt.conack.flags')
        # so string domain names like 'raspberrypi.local' become NaN instead of erroring.
        for col in self.cont_features:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        result_df = result_df.dropna()
        logger.debug(f"[+] Cleaned mixed-type & missing rows: {len(result_df):,} rows remain")
        return result_df

    def setup_encoders(self, df: pd.DataFrame):
        self.encoders = {
            'categorical': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'minmax': MinMaxScaler(),
            'quantile_uniform': QuantileTransformer(output_distribution='normal', random_state=42),
            'label': LabelEncoder(),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
            'standard': StandardScaler()
        }
        # Cast categorical columns to string to handle mixed float/string values
        # (e.g. '0.0' placeholder + real strings like 'GET', 'MQTT')
        cat_df = df[self.encoded_categorical_features].astype(str)
        self.encoders['categorical'].fit(cat_df)
        self.encoders['minmax'].fit(df[self.encoded_numerical_features])
        self.encoders['quantile_uniform'].fit(df[self.encoded_numerical_features])
        self.encoders['standard'].fit(df[self.encoded_numerical_features])
        self.encoders['label'].fit(df[self.label_column])
        self.encoders['ordinal'].fit(cat_df)

    def load_encoders(self, encoders_dir=None, **kwargs):
        """Load pre-trained encoders from saved files"""
        import joblib
        if encoders_dir is None:
            encoders_dir = self.encoders_dir
        try:
            self.encoders = {
                'categorical': joblib.load(f"{encoders_dir}/categorical_encoder.pkl"),
                'minmax': joblib.load(f"{encoders_dir}/minmax_encoder.pkl"),
                'quantile_uniform': joblib.load(f"{encoders_dir}/quantile_uniform_encoder.pkl"),
                'label': joblib.load(f"{encoders_dir}/label_encoder.pkl"),
                'ordinal': joblib.load(f"{encoders_dir}/ordinal_encoder.pkl"),
                'standard': joblib.load(f"{encoders_dir}/standard_encoder.pkl")
            }
            logger.info(f"[+] Encoders loaded from {encoders_dir}")
            return True
        except FileNotFoundError as e:
            logger.error(f"[-] Encoder files not found in {encoders_dir}: {e}")
            return False
        except Exception as e:
            logger.error(f"[-] Error loading encoders: {e}")
            return False

    def save_encoders(self, encoders_dir=None):
        """Save trained encoders to files"""
        import joblib
        import os
        
        if encoders_dir is None:
            encoders_dir = self.encoders_dir
            
        os.makedirs(encoders_dir, exist_ok=True)
        
        try:
            joblib.dump(self.encoders['categorical'], f"{encoders_dir}/categorical_encoder.pkl")
            joblib.dump(self.encoders['minmax'], f"{encoders_dir}/minmax_encoder.pkl")
            joblib.dump(self.encoders['label'], f"{encoders_dir}/label_encoder.pkl")
            joblib.dump(self.encoders['ordinal'], f"{encoders_dir}/ordinal_encoder.pkl")
            joblib.dump(self.encoders['quantile_uniform'], f"{encoders_dir}/quantile_uniform_encoder.pkl")
            joblib.dump(self.encoders['standard'], f"{encoders_dir}/standard_encoder.pkl")
            logger.info(f"[+] Encoders saved to {encoders_dir}")
            return True
        except Exception as e:
            logger.error(f"[-] Error saving encoders: {e}")
            return False

    def label_distribution(self, df: pd.DataFrame):
        return df[self.label_column].value_counts()
    
    def preprocess_encode_categorical_features(self, df: pd.DataFrame):
        df = df.copy()
        encoder = self.encoders['categorical']
        cat_df = df[self.encoded_categorical_features].astype(str)
        encoded_train = encoder.transform(cat_df)
        encoded_train_df = pd.DataFrame(
            encoded_train, 
            columns=encoder.get_feature_names_out(self.encoded_categorical_features),
            index=df.index
        )
        df = pd.concat([encoded_train_df, df], axis=1)
        df = df.drop(columns=self.encoded_categorical_features)
        return df

    def preprocess_encode_numerical_features_minmax(self, df: pd.DataFrame):
        df = df.copy()
        scaler = self.encoders['minmax']
        df[self.encoded_numerical_features] = scaler.transform(df[self.encoded_numerical_features])
        return df

    def preprocess_encode_numerical_features_standard(self, df: pd.DataFrame, **kwargs):
        df_clean = df.copy()
        scaler = self.encoders['standard']
        df_clean[self.encoded_numerical_features] = scaler.transform(df_clean[self.encoded_numerical_features])
        return df_clean

    def preprocess_encode_numerical_features_quantile_uniform(self, df: pd.DataFrame):
        df_clean = df.copy()
        scaler = self.encoders['quantile_uniform']
        df_clean[self.encoded_numerical_features] = scaler.transform(df_clean[self.encoded_numerical_features])
        return df_clean

    def preprocess_encode_ordinal_features(self, df: pd.DataFrame):
        df = df.copy()
        encoder = self.encoders['ordinal']
        cat_df = df[self.encoded_categorical_features].astype(str)
        df[self.encoded_categorical_features] = encoder.transform(cat_df)
        df[self.encoded_categorical_features] = df[self.encoded_categorical_features].astype(int)
        return df
    
    def inverse_transform_ordinal_features(self, df: pd.DataFrame):
        df = df.copy()
        encoder = self.encoders['ordinal']
        df[self.encoded_categorical_features] = encoder.inverse_transform(df[self.encoded_categorical_features])
        return df

    def preprocess_encode_binary_features(self, df: pd.DataFrame):
        """Encode binary features - keep them as 0/1"""
        df = df.copy()
        # Binary features should remain as 0/1, no scaling needed
        for feature in self.binary_features:
            if feature in df.columns:
                # Ensure binary values (0 or 1)
                df[feature] = df[feature].astype(int)
        return df

    def preprocess_encode_label(self, df: pd.DataFrame):
        df = df.copy()
        encoder = self.encoders['label']
        df[self.label_column] = encoder.transform(df[self.label_column])
        return df

    def inverse_transform_label(self, df: pd.DataFrame):
        if 'label' in self.encoders and self.label_column in df.columns:
            label_encoder = self.encoders['label']
            df[self.label_column] = label_encoder.inverse_transform(df[self.label_column])
        return df

    def inverse_transform(self, df: pd.DataFrame, numerical_inverse: str = 'minmax'):
        """
        Inverse transform the encoded DataFrame back to original format
        """
            
        df_inverse = df.copy()

        if len(df_inverse) == 0:
            logger.debug("[+] Inverse transform skipped for empty dataframe")
            return pd.DataFrame(columns=self.features + [self.label_column], index=df_inverse.index)
        
        # 1. Inverse transform categorical features (one-hot -> original strings)
        if 'categorical' in self.encoders:
            encoder = self.encoders['categorical']
            expected_onehot_columns = list(encoder.get_feature_names_out(self.encoded_categorical_features))
            onehot_columns = [col for col in expected_onehot_columns if col in df_inverse.columns]

            if onehot_columns:
                # Rebuild full one-hot matrix in fitted column order for robust inverse transform.
                onehot_data_df = pd.DataFrame(0.0, index=df_inverse.index, columns=expected_onehot_columns)
                onehot_data_df[onehot_columns] = df_inverse[onehot_columns]

                if len(onehot_columns) < len(expected_onehot_columns):
                    missing_onehot = len(expected_onehot_columns) - len(onehot_columns)
                    logger.warning(f"[!] Missing {missing_onehot} one-hot columns for inverse transform; filling with 0")

                original_cat = encoder.inverse_transform(onehot_data_df.values)
                original_cat_df = pd.DataFrame(
                    original_cat,
                    columns=self.encoded_categorical_features,
                    index=df_inverse.index,
                )

                # Replace one-hot columns with original categorical columns
                df_inverse = df_inverse.drop(columns=onehot_columns)
                df_inverse = pd.concat([df_inverse, original_cat_df], axis=1)
                logger.debug("[+] Inverse transformed categorical features")
        
        # 2. Inverse transform numerical features (scaled -> original values)
        # Find which numerical features exist in current df
        existing_numerical = [col for col in self.encoded_numerical_features if col in df_inverse.columns]
        if numerical_inverse == 'minmax' and 'minmax' in self.encoders:
            scaler = self.encoders['minmax']
            if existing_numerical:
                df_inverse[existing_numerical] = scaler.inverse_transform(df_inverse[existing_numerical])
                logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via MinMaxScaler")
        elif numerical_inverse == 'standard' and 'standard' in self.encoders:
            scaler = self.encoders['standard']
            if existing_numerical:
                df_inverse[existing_numerical] = scaler.inverse_transform(df_inverse[existing_numerical])
                logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via StandardScaler")
        elif numerical_inverse == 'quantile_uniform' and 'quantile_uniform' in self.encoders:
            scaler = self.encoders['quantile_uniform']
            if existing_numerical:
                df_inverse[existing_numerical] = scaler.inverse_transform(df_inverse[existing_numerical])
                logger.debug(f"[+] Inverse transformed {len(existing_numerical)} numerical features via QuantileTransformer")
        
        # 3. Inverse transform labels (encoded -> original strings)
        if 'label' in self.encoders and self.label_column in df_inverse.columns:
            label_encoder = self.encoders['label']
            df_inverse[self.label_column] = label_encoder.inverse_transform(df_inverse[self.label_column])
            logger.debug(f"[+] Inverse transformed labels")
        
        # 4. Fix binary features - convert back to 0/1
        for feature in self.binary_features:
            if feature in df_inverse.columns:
                # Convert to binary using threshold 0.5
                df_inverse[feature] = (df_inverse[feature] > 0.5).astype(int)
                logger.debug(f"[+] Fixed binary feature {feature}: converted to 0/1")
        
        # 4b. Clamp tiny numerical magnitudes and enforce non-negativity on numerical features
        try:
            epsilon = 1e-9
            num_cols_present = [col for col in self.cont_features if col in df_inverse.columns]
            if num_cols_present:
                # Set |x| < epsilon to 0 and negatives to 0
                df_inverse[num_cols_present] = df_inverse[num_cols_present].mask(df_inverse[num_cols_present].abs() < epsilon, 0.0)
                df_inverse[num_cols_present] = df_inverse[num_cols_present].mask(df_inverse[num_cols_present] < 0, 0.0)
                logger.debug(f"[+] Clamped tiny and negative values to 0 for {len(num_cols_present)} numerical features")
                # Soft rounding to reduce floating noise without being too strict
                rounding_decimals = 8
                df_inverse[num_cols_present] = df_inverse[num_cols_present].round(rounding_decimals)
                logger.debug(f"[+] Rounded numerical features to {rounding_decimals} decimals")
        except Exception as e:
            logger.warning(f"[!] Failed to clamp tiny/negative numerical values: {e}")

        
        # 5. Reorder columns to match original order
        original_order = self.features + [self.label_column]

        missing_after_inverse = [col for col in self.encoded_categorical_features if col not in df_inverse.columns]
        if missing_after_inverse:
            logger.warning(f"[!] Missing categorical columns after inverse transform: {missing_after_inverse}")
        
        # Only keep columns that exist
        existing_columns = [col for col in original_order if col in df_inverse.columns]
        df_inverse = df_inverse[existing_columns]
        
        logger.debug(f"[+] Inverse transform completed. Shape: {df_inverse.shape}")
        return df_inverse

    def extract_categorical_cardinalities(self):
        features_and_cardinalities = {}
        features = self.encoders['ordinal'].feature_names_in_
        categories = self.encoders['ordinal'].categories_
        for feature, category in zip(features, categories):
            safe_name = feature.replace(".", "_")
            features_and_cardinalities[safe_name] = len(category)
            logger.debug(f"[+] Cardinality of {feature} ({safe_name}): {len(category)}")
        return features_and_cardinalities

