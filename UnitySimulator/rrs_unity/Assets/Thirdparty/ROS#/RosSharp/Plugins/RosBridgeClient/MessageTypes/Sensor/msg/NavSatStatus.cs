/* 
 * This message is auto generated by ROS#. Please DO NOT modify.
 * Note:
 * - Comments from the original code will be written in their own line 
 * - Variable sized arrays will be initialized to array of size 0 
 * Please report any issues at 
 * <https://github.com/siemens/ros-sharp> 
 */

namespace RosSharp.RosBridgeClient.MessageTypes.Sensor
{
    public class NavSatStatus : Message
    {
        public const string RosMessageName = "sensor_msgs/NavSatStatus";

        //  Navigation Satellite fix status for any Global Navigation Satellite System
        //  Whether to output an augmented fix is determined by both the fix
        //  type and the last time differential corrections were received.  A
        //  fix is valid when status >= STATUS_FIX.
        public const sbyte STATUS_NO_FIX = -1; //  unable to fix position
        public const sbyte STATUS_FIX = 0; //  unaugmented fix
        public const sbyte STATUS_SBAS_FIX = 1; //  with satellite-based augmentation
        public const sbyte STATUS_GBAS_FIX = 2; //  with ground-based augmentation
        public sbyte status { get; set; }
        //  Bits defining which Global Navigation Satellite System signals were
        //  used by the receiver.
        public const ushort SERVICE_GPS = 1;
        public const ushort SERVICE_GLONASS = 2;
        public const ushort SERVICE_COMPASS = 4; //  includes BeiDou.
        public const ushort SERVICE_GALILEO = 8;
        public ushort service { get; set; }

        public NavSatStatus()
        {
            this.status = 0;
            this.service = 0;
        }

        public NavSatStatus(sbyte status, ushort service)
        {
            this.status = status;
            this.service = service;
        }
    }
}
