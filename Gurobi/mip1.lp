\ Model mip1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  s_[0] + s_[1] + s_[2] + s_[3] + s_[4] + 0 s_[5] + 0 s_[6] + 0 s_[7]
   + 0 s_[8] + 0 s_[9] + 0 re_x_[0] + 0 re_x_[1] + 0 re_x_[2] + 0 re_x_[3]
   + 0 re_x_[4] + 0 hit_[0] + 0 hit_[1] + 0 hit_[2] + 0 hit_[3] + 0 hit_[4]
   + 0 g_[0] + 0 g_[1] + 0 g_[2] + 0 g_[3] + 0 g_[4]
Subject To
 R0: a >= 3
 R1: s_[0] + s_[1] + s_[2] + s_[3] + s_[4] <= 3
 re_v1_[0]: s_[0] = 0 -> re_x_[0] = 1
 re_v2_[0]: s_[0] = 1 -> re_x_[0] = 0
 re_v1_[1]: s_[1] = 0 -> re_x_[1] = 1
 re_v2_[1]: s_[1] = 1 -> re_x_[1] = 0
 re_v1_[2]: s_[2] = 0 -> re_x_[2] = 1
 re_v2_[2]: s_[2] = 1 -> re_x_[2] = 0
 re_v1_[3]: s_[3] = 0 -> re_x_[3] = 1
 re_v2_[3]: s_[3] = 1 -> re_x_[3] = 0
 re_v1_[4]: s_[4] = 0 -> re_x_[4] = 1
 re_v2_[4]: s_[4] = 1 -> re_x_[4] = 0
Bounds
 2 <= a <= 5
Binaries
 s_[0] s_[1] s_[2] s_[3] s_[4] s_[5] s_[6] s_[7] s_[8] s_[9] re_x_[0]
 re_x_[1] re_x_[2] re_x_[3] re_x_[4] hit_[0] hit_[1] hit_[2] hit_[3]
 hit_[4] g_[0] g_[1] g_[2] g_[3] g_[4]
Generals
 a
End
