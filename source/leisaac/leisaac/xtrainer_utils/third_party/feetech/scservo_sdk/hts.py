#!/usr/bin/env python
#  寄存器表 ： http://ftdoc.longbos.com/#/12_HTS3235_13
# date :2024-03-07
from .scservo_def import *
from .protocol_packet_handler import *
from .group_sync_write import *
from .group_sync_read import *
import numpy as np
# 力矩输出与否
HTS_TORQUE_ENABLE = 1
HTS_TORQUE_DISABLE = 0

#波特率定义
HTS_1M = 0
HTS_0_5M = 1
HTS_250K = 2
HTS_128K = 3
HTS_115200 = 4
HTS_76800 = 5
HTS_57600 = 6
HTS_38400 = 7

#内存表定义
#-------EPROM(只读)--------
HTS_MODEL_L = 3
HTS_MODEL_H = 4

#-------EPROM(读写)--------
hts_id = 5
HTS_BAUD_RATE = 6
HTS_MIN_ANGLE_LIMIT_L = 9
HTS_MIN_ANGLE_LIMIT_H = 10
HTS_MAX_ANGLE_LIMIT_L = 11
HTS_MAX_ANGLE_LIMIT_H = 12
HTS_CW_DEAD = 26
HTS_CCW_DEAD = 27


HTS_WORK_MODE = 33 # 工作模式

#-------SRAM(读写)--------
HTS_TORQUE_ENABLE_ADDR = 40
HTS_ACC = 41 # HTS  新增 加速度字段
HTS_GOAL_POSITION_L = 42
HTS_GOAL_POSITION_H = 43
HTS_GOAL_TORQUE_L = 44
HTS_GOAL_TORQUE_H = 45
HTS_GOAL_SPEED_L = 46
HTS_GOAL_SPEED_H = 47
HTS_LOCK = 48  # 转矩限制

#-------SRAM(只读)--------
HTS_PRESENT_POSITION_L  = 56
HTS_PRESENT_POSITION_H = 57
HTS_PRESENT_SPEED_L = 58
HTS_PRESENT_SPEED_H = 59
HTS_PRESENT_LOAD_L = 60
HTS_PRESENT_LOAD_H = 61
HTS_PRESENT_VOLTAGE = 62
HTS_PRESENT_TEMPERATURE = 63
HTS_IS_ASYNC_WRITE = 64
HTS_PRESENT_STATUS = 65
HTS_MOVING = 66
HTS_PRESENT_CURRENT_L = 69
HTS_PRESENT_CURRENT_H = 70

class HTS(protocol_packet_handler):
    def __init__(self, portHandler,id_list,packet_handler):
        print("HTS __init__ !!!!!!")
        self.portHandler = portHandler
        self.packet_handler = packet_handler
        self.groupSyncWrite = GroupSyncWrite(
            self.packet_handler,
            HTS_TORQUE_ENABLE_ADDR,
            HTS_GOAL_SPEED_H - HTS_TORQUE_ENABLE_ADDR + 1
        )
        self.id_list = id_list
        self.groupSyncRead = GroupSyncRead(
            self.packet_handler,            
            HTS_PRESENT_POSITION_L,
            HTS_PRESENT_CURRENT_H - HTS_PRESENT_POSITION_L + 1,
        )        
        super().__init__(self.portHandler,1) # HTS 和 SCS相似，使用 1 类型的协议，这里的0 和1 负责字节数据的翻转
        for id in self.id_list:
            print("init  ... ",id)
            if not self.groupSyncRead.addParam(id):
                raise RuntimeError(
                    f"Failed to add parameter for feetech with ID {id}"
                )
        # data = np.zeros(HTS_GOAL_SPEED_H - HTS_TORQUE_ENABLE_ADDR + 1,dtype=int)
        # for id in self.id_list:
        #     print("init  ... ",id)
        #     if not self.groupSyncWrite.addParam(id,data):
        #         raise RuntimeError(
        #             f"Failed to add parameter for feetech with ID {id}"
        #         )
    # def WritePos(self, hts_id, position, time, speed):
    #     txpacket = [self.scs_lobyte(position), self.scs_hibyte(position), self.scs_lobyte(time), self.scs_hibyte(time), self.scs_lobyte(speed), self.scs_hibyte(speed)]
    #     return self.writeTxRx(hts_id, HTS_GOAL_POSITION_L, len(txpacket), txpacket)

    # def ReadPos(self, hts_id):
    #     scs_present_position, scs_comm_result, scs_error = self.read2ByteTxRx(hts_id, HTS_PRESENT_POSITION_L)
    #     return scs_present_position, scs_comm_result, scs_error

    # def ReadSpeed(self, hts_id):
    #     scs_present_speed, scs_comm_result, scs_error = self.read2ByteTxRx(hts_id, HTS_PRESENT_SPEED_L)
    #     return self.scs_tohost(scs_present_speed, 15), scs_comm_result, scs_error

    # def ReadPosSpeed(self, hts_id):
    #     scs_present_position_speed, scs_comm_result, scs_error = self.read4ByteTxRx(hts_id, HTS_PRESENT_POSITION_L)
    #     scs_present_position = self.scs_loword(scs_present_position_speed)
    #     scs_present_speed = self.scs_hiword(scs_present_position_speed)
    #     return scs_present_position, self.scs_tohost(scs_present_speed, 15), scs_comm_result, scs_error

    # def ReadMoving(self, hts_id):
    #     moving, scs_comm_result, scs_error = self.read1ByteTxRx(hts_id, HTS_MOVING)
    #     return moving, scs_comm_result, scs_error

    def u16_to_s16(self,unsigned_int):
        if unsigned_int & 0x8000:  # 检查最高位（bit15）是否为1            
            signed_int = (unsigned_int - 0x8000) * -1     # feetech 的特有的转换形式
        else:
            # 如果bit15为0，则表示为正数，无需转换
            signed_int = unsigned_int
        return signed_int
    # def SyncWritePos(self, hts_id, position, time, speed):
    #     txpacket = [self.scs_lobyte(position), self.scs_hibyte(position), self.scs_lobyte(time), self.scs_hibyte(time), self.scs_lobyte(speed), self.scs_hibyte(speed)]
    #     return self.groupSyncWrite.addParam(hts_id, txpacket)
    # 顺序写入所有的数据到预发送区域 
    # enable_switch : 是否使能力矩模式 0：不开启 1：打开力矩输出 2:打开阻尼
    # acc: 加速度 范围是 0~ 254 0 是最大 单位是1.46RPM/s
    # pos: 目标位置 
    # torqye: 目标电流 
    # speed 目标速度
    def SyncWriteAll(self,id_list, enable_switch_list,acc_list,pos_list,torque_list,speed_list):
        if len(id_list) != len(enable_switch_list)  or\
            len(id_list) != len(acc_list) or\
            len(id_list) != len(pos_list) or\
            len(id_list) != len(torque_list) or\
            len(id_list) != len(speed_list) :
                raise ValueError(
                " ERROR  input !!!! ",id_list, enable_switch_list,acc_list,pos_list,torque_list,speed_list
            ) 
        isErr = False
        self.groupSyncWrite.clearParam()
        # for id,en,acc,pos,tor,spd  in zip(id_list, enable_switch_list,acc_list,pos_list,torque_list,speed_list):
        for i, id in enumerate(id_list):
            txpacket = [
                enable_switch_list[i],acc_list[i], # enable: 1byte ,acc 1byte 
                self.scs_lobyte(pos_list[i]), self.scs_hibyte(pos_list[i]),          # pos 2byte
                self.scs_lobyte(torque_list[i]), self.scs_hibyte(torque_list[i]),    # torque 2byte
                self.scs_lobyte(speed_list[i]), self.scs_hibyte(speed_list[i])       # speed  2byte
            ]
            # print("i = ",i,";id = ",id,"tx = ",txpacket)
            res =  self.groupSyncWrite.addParam(id, txpacket)        
            isErr |= not res
            if isErr == True:
                print("id : ",id,";isErr = ",isErr,";res = ",res)
        if isErr == False:
            hts_comm_result = self.groupSyncWrite.txPacket()
            if hts_comm_result != COMM_SUCCESS:
                print("%s" % self.portHandler.getTxRxResult(hts_comm_result))
                isErr = True
        else: 
            print("isErr = ",isErr)
        return isErr
    def SyncReadAll(self):
        _joint_angles   = np.zeros(len(self.id_list), dtype=float)
        _joint_speed    = np.zeros(len(self.id_list), dtype=float)
        _joint_load     = np.zeros(len(self.id_list), dtype=float)
        _joint_voltage  = np.zeros(len(self.id_list), dtype=float)
        _joint_temp     = np.zeros(len(self.id_list), dtype=float)
        _joint_status   = np.zeros(len(self.id_list), dtype=float)
        _joint_isMoving = np.zeros(len(self.id_list), dtype=float)
        _joint_current  = np.zeros(len(self.id_list), dtype=float)
        result = self.groupSyncRead.txRxPacket()
        if result != COMM_SUCCESS:
            print(f"warning, comm failed: {result}")
            return False
        ######################## begin to get data from servo ###############################
        for i, id in enumerate(self._ids):
            # print("i,id = ",i,id)
            if self.groupSyncRead.isAvailable(
                i, HTS_PRESENT_POSITION_L, HTS_PRESENT_CURRENT_H - HTS_PRESENT_POSITION_L + 1
            ):
                angle = self.groupSyncRead.getData(id, HTS_PRESENT_POSITION_L, 2)
                    # 根据手册得来的转换公式 不特殊说明，下面的转换公式均来自 http://ftdoc.longbos.com/#/11_HTS3032_10                
                angle = self.u16_to_s16(angle) # 转有符号数据                
                _joint_angles[i] = angle * 0.087
                speed = self.groupSyncRead.getData(id, HTS_PRESENT_SPEED_L, 2)
                _joint_speed[i] = speed * 0.732 # 单位是 RPM                        
                load = self.groupSyncRead.getData(id, HTS_PRESENT_LOAD_L, 2)
                _joint_load[i] = load * 0.001
                voltage = self.groupSyncRead.getData(id, HTS_PRESENT_VOLTAGE, 1)
                _joint_voltage[i] = voltage*0.1
                temperature = self.groupSyncRead.getData(id, HTS_PRESENT_TEMPERATURE, 1)
                _joint_temp[i] = temperature 
                status = self.groupSyncRead.getData(id, HTS_PRESENT_STATUS, 1)
                _joint_status[i] = status 
                _isMoving = self.groupSyncRead.getData(id, HTS_MOVING, 1)
                _joint_isMoving[i] = _isMoving
                current = self.groupSyncRead.getData(id, HTS_PRESENT_CURRENT_L, 2)
                current = self.u16_to_s16(current) # 转有符号数据
                _joint_current[i] = current * 6.5 #
            else:
                raise RuntimeError(
                    f"Failed to get joint angles for feetech with ID {i}"
                )
            # print("_joint_current",_joint_current)
        return _joint_angles,_joint_speed,_joint_load,_joint_voltage,_joint_temp,_joint_status,_joint_isMoving,_joint_current
        ######################## finish get data from servo ###############################
    
    def RegWritePos(self, hts_id, position, time, speed):
        txpacket = [self.scs_lobyte(position), self.scs_hibyte(position), self.scs_lobyte(time), self.scs_hibyte(time), self.scs_lobyte(speed), self.scs_hibyte(speed)]
        return self.regWriteTxRx(hts_id, HTS_GOAL_POSITION_L, len(txpacket), txpacket)

    def RegAction(self):
        return self.action(BROADCAST_ID)

    def LockEprom(self, hts_id):
        return self.write1ByteTxRx(hts_id, HTS_LOCK, 1)

    def unLockEprom(self, hts_id):
        return self.write1ByteTxRx(hts_id, HTS_LOCK, 0)

