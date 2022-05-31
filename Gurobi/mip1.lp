\ Model mip1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  a + s_[0] + s_[1] + s_[2] + s_[3] + s_[4] + s_[5] + s_[6] + s_[7] + s_[8]
   + s_[9]
Subject To
 a: a >= 3
 a2: a <= 2
Bounds
 a <= 1
Binaries
 s_[0] s_[1] s_[2] s_[3] s_[4] s_[5] s_[6] s_[7] s_[8] s_[9]
Generals
 a
End
