import enum
import logging
from livekit.agents import llm
from typing import Annotated

logger = logging.getLogger("tv-controller")
logger.setLevel(logging.INFO)

class TVAction(enum.Enum):
    POWER_ON = "power_on"
    POWER_OFF = "power_off"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MUTE = "mute"
    UNMUTE = "unmute"
    NEXT_CHANNEL = "next_channel"
    PREVIOUS_CHANNEL = "previous_channel"
    DIRECT_CHANNEL_SELECTION = "direct_channel_selection"

class TVController(llm.FunctionContext):
    @llm.ai_callable(description="Activate the television")
    def power_on(self):
        logger.info("Power On command received")
        return "Television: Power On"

    @llm.ai_callable(description="Deactivate the television")
    def power_off(self):
        logger.info("Power Off command received")
        return "Television: Power Off"

    @llm.ai_callable(description="Increase the volume by one unit")
    def volume_up(self):
        logger.info("Volume Up command received")
        return "Television: Volume Up"

    @llm.ai_callable(description="Decrease the volume by one unit")
    def volume_down(self):
        logger.info("Volume Down command received")
        return "Television: Volume Down"

    @llm.ai_callable(description="Mute the television")
    def mute(self):
        logger.info("Mute command received")
        return "Television: Mute"

    @llm.ai_callable(description="Unmute the television")
    def unmute(self):
        logger.info("Unmute command received")
        return "Television: Unmute"

    @llm.ai_callable(description="Advance to the subsequent channel")
    def next_channel(self):
        logger.info("Next Channel command received")
        return "Television: Next Channel"

    @llm.ai_callable(description="Revert to the previous channel")
    def previous_channel(self):
        logger.info("Previous Channel command received")
        return "Television: Previous Channel"

    @llm.ai_callable(description="Switch to a specified channel number")
    def direct_channel_selection(self, channel: Annotated[int, llm.TypeInfo(description="The channel number to select")]):
        logger.info("Direct Channel Selection command received for channel: %s", channel)
        return f"Television: Direct Channel Selection to {channel}"
