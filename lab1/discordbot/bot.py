import os
import asyncio

from workflow import octoshop_workflow
from discord import Intents, Attachment
from discord.ext import commands
from concurrent.futures import ThreadPoolExecutor

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
DISCORD_COMMAND = os.environ["DISCORD_COMMAND"]


intents = Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="/",
    description="Generate beautiful images",
    intents=intents,
)


@bot.event
async def on_ready():
    print("Bot is Up and Ready!")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)!")
    except Exception as e:
        print(e)


@bot.tree.command(name=DISCORD_COMMAND, description="Generate beautiful images")
async def generate(interaction, file: Attachment):
    try:
        # Defer right away cause workflow might take more than one second.
        # Also, set ephemeral to False, so generations are publicly available
        await interaction.response.defer(ephemeral=False)
        # Create an async event loop to execute a the blocking function (octoshop workflow)
        loop = asyncio.get_event_loop()
        output, clip_output, llama2_output = await loop.run_in_executor(
            ThreadPoolExecutor(), octoshop_workflow, file.url
        )
        # Format output text
        content = "CLIP Interrogator output: {}\nLLAMA2 output: {}".format(
            clip_output, llama2_output
        )
        # Convert Discord Attachment into Discord File
        input = await file.to_file()
        # Send output text and input/output image files back
        await interaction.followup.send(content=content, files=[input, output])
    except Exception as e:
        print(f"Error: {e}")
        await interaction.response.send_message("Generation failed, please try again!")


bot.run(DISCORD_TOKEN)
