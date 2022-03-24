/* 
 * This message is auto generated by ROS#. Please DO NOT modify.
 * Note:
 * - Comments from the original code will be written in their own line 
 * - Variable sized arrays will be initialized to array of size 0 
 * Please report any issues at 
 * <https://github.com/siemens/ros-sharp> 
 */

namespace RosSharp.RosBridgeClient.MessageTypes.Std
{
    public class UInt8 : Message
    {
        public const string RosMessageName = "std_msgs/UInt8";

        public byte data { get; set; }

        public UInt8()
        {
            this.data = 0;
        }

        public UInt8(byte data)
        {
            this.data = data;
        }
    }
}
