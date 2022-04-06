def output_action(self, action_in):
    action_out = [0.0] * len(action_in)
    count = 0
    download = 0
    wait_line = []
    d_full = False
    for i in reversed(range(len(action_in))):  # -1, -2 ....
        if count >= self.B_size:  #
            return action_out
        else:
            if download >= self.D_size:
                # still cache
                if self.t_minus_s_[i] == 1 and action_in[i] > 0.0:  # cache still
                    action_out[i] = 1
                    count += 1
            else:
                # download new stuff
                if action_in[i] > 0.0:
                    if self.t_minus_s_[i] == 1:  # cache still
                        action_out[i] = 1
                        count += 1
                    else:  # download
                        action_out[i] = 1
                        count += 1
                        download += 1

    return action_out