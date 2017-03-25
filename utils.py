names = ["duration", "protocol_type", "service", "flag", "src_bytes",
         "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
         "num_failed_logins", "logged_in", "num_compromised", "root_shell",
         "su_attempted", "num_root", "num_file_creations", "num_shells",
         "num_access_files", "num_outbound_cmds", "is_host_login",
         "is_guest_login", "count", "srv_count", "serror_rate",
         "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
         "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
         "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
         "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
         "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
         "dst_host_srv_serror_rate", "dst_host_rerror_rate",
         "dst_host_srv_rerror_rate", "label"]

categorical_names = ['protocol_type', 'service', 'flag']
label_names = ['label', 'label_binary_num', 'label_four', 'label_four_num']
names_without_changes = ['num_outbound_cmds', 'is_host_login']

names_to_normalize = ["duration", "src_bytes", "dst_bytes", "wrong_fragment",
                      "urgent", "hot", "num_failed_logins", "num_compromised",
                      "num_root", "num_file_creations", "num_shells",
                      "num_access_files", "num_outbound_cmds"]

label_to_four_attack_class = {'back.': 'DOS',
                              'land.': 'DOS',
                              'neptune.': 'DOS',
                              'pod.': 'DOS',
                              'smurf.': 'DOS',
                              'teardrop.': 'DOS',
                              'satan.': 'Probe',
                              'ipsweep.': 'Probe',
                              'nmap.': 'Probe',
                              'portsweep.': 'Probe',
                              'normal.': 'Normal',
                              'guess_passwd.': 'R2L',
                              'ftp_write.': 'R2L',
                              'imap.': 'R2L',
                              'phf.': 'R2L',
                              'spy.': 'R2L',
                              'multihop.': 'R2L',
                              'warezmaster.': 'R2L',
                              'warezclient.': 'R2L',
                              'buffer_overflow.': 'U2R',
                              'loadmodule.': 'U2R',
                              'perl.': 'U2R',
                              'rootkit.': 'U2R'}

five_classes_to_num = {'Normal': 0,
                       'Probe': 1,
                       'R2L': 2,
                       'U2R': 3,
                       'DOS': 4}

# Traffic features computed using a two-second time window
traffic_features = ["count", "srv_count", "serror_rate", "srv_serror_rate",
                    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                    "dst_host_srv_count", "dst_host_same_srv_rate",
                    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                    "dst_host_srv_rerror_rate"]

my_colors = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'pink', 'indigo', 'crimson',
             'gray', 'olivedrab', 'orange', 'maroon', 'magenta', 'violet',
             'yellowgreen', 'lime', 'lightyellow', 'lavender', 'indigo'] * 5
