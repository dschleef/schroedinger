
.function orc_add2_rshift_add_s16_22
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.temp 2 t1

addw t1, s1, s2
addw t1, t1, 2
shrsw t1, t1, 2
addw d1, d1, t1

.function orc_add2_rshift_sub_s16_22
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.temp 2 t1

addw t1, s1, s2
addw t1, t1, 2
shrsw t1, t1, 2
subw d1, d1, t1

.function orc_add2_rshift_add_s16_11
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.temp 2 t1

avgsw t1, s1, s2
#addw t1, s1, s2
#addw t1, t1, 1
#shrsw t1, t1, 1
addw d1, d1, t1

.function orc_add2_rshift_sub_s16_11
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.temp 2 t1

avgsw t1, s1, s2
#addw t1, s1, s2
#addw t1, t1, 1
#shrsw t1, t1, 1
subw d1, d1, t1

.function orc_add_const_rshift_s16_11
.dest 2 d1 int16_t
.source 2 s1 int16_t
.temp 2 t1

addw t1, s1, 1
shrsw d1, t1, 1


.function orc_add_s16
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t

addw d1, s1, s2


.function orc_addc_rshift_s16
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.temp 2 t1
.param 2 p1

addw t1, s1, s2
shrsw d1, t1, p1


.function orc_lshift1_s16
.dest 2 d1 int16_t
.source 2 s1 int16_t

shlw d1, s1, 1


.function orc_lshift2_s16
.dest 2 d1 int16_t
.source 2 s1 int16_t

shlw d1, s1, 2


.function orc_lshift_s16_ip
.dest 2 d1 int16_t
.param 2 p1

shlw d1, d1, p1


.function orc_mas2_add_s16_ip
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.temp 2 t1
.temp 4 t2
.param 2 p1
.param 4 p2
.param 4 p3

addw t1, s1, s2
mulswl t2, t1, p1
addl t2, t2, p2
shrsl t2, t2, p3
convlw t1, t2
addw d1, d1, t1


.function orc_mas2_sub_s16_ip
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.temp 2 t1
.temp 4 t2
.param 2 p1
.param 4 p2
.param 4 p3

addw t1, s1, s2
mulswl t2, t1, p1
addl t2, t2, p2
shrsl t2, t2, p3
convlw t1, t2
subw d1, d1, t1


.function orc_mas4_across_add_s16_1991_ip
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.source 2 s3 int16_t
.source 2 s4 int16_t
.param 4 p1
.param 4 p2
.temp 2 t1
.temp 2 t2
.temp 4 t3
.temp 4 t4

addw t1, s2, s3
mulswl t3, t1, 9
addw t2, s1, s4
convswl t4, t2
subl t3, t3, t4
addl t3, t3, p1
shrsl t3, t3, p2
convlw t1, t3
addw d1, d1, t1


.function orc_mas4_across_sub_s16_1991_ip
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t
.source 2 s3 int16_t
.source 2 s4 int16_t
.param 4 p1
.param 4 p2
.temp 2 t1
.temp 2 t2
.temp 4 t3
.temp 4 t4

addw t1, s2, s3
mulswl t3, t1, 9
addw t2, s1, s4
convswl t4, t2
subl t3, t3, t4
addl t3, t3, p1
shrsl t3, t3, p2
convlw t1, t3
subw d1, d1, t1


.function orc_subtract_s16
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 2 s2 int16_t

subw d1, s1, s2


.function orc_memcpy
.dest 1 d1 void
.source 1 s1 void

copyb d1, s1


.function orc_add_s16_u8
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 1 s2
.temp 2 t1

convubw t1, s2
addw d1, t1, s1


.function orc_convert_s16_u8
.dest 2 d1
.source 1 s1

convubw d1, s1


.function orc_convert_u8_s16
.dest 1 d1
.source 2 s1 int16_t

convsuswb d1, s1


.function orc_subtract_s16_u8
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 1 s2
.temp 2 t1

convubw t1, s2
subw d1, s1, t1


.function orc_multiply_and_add_s16_u8
.dest 2 d1 int16_t
.source 2 s1 int16_t
.source 1 s2
.temp 2 t1

convubw t1, s2
mullw t1, t1, s1
addw d1, d1, t1


.function orc_splat_s16_ns
.dest 2 d1 int16_t
.param 2 p1

copyw d1, p1


.function orc_splat_u8_ns
.dest 1 d1
.param 1 p1

copyb d1, p1


.function orc_average_u8
.dest 1 d1
.source 1 s1
.source 1 s2

avgub d1, s1, s2


.function orc_rrshift6_s16_ip
.dest 2 d1 int16_t
.temp 2 t1

subw t1, d1, 8160
shrsw d1, t1, 6


.function orc_unpack_yuyv_y
.dest 1 d1
.source 2 s1

select0wb d1, s1


.function orc_unpack_yuyv_u
.dest 1 d1
.source 4 s1
.temp 2 t1

select0lw t1, s1
select1wb d1, t1


.function orc_unpack_yuyv_v
.dest 1 d1
.source 4 s1
.temp 2 t1

select1lw t1, s1
select1wb d1, t1


