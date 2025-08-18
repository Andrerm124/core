"""Conversation support for the Google Generative AI Conversation integration."""

from __future__ import annotations

import codecs
from collections.abc import AsyncGenerator, Callable
from dataclasses import replace
from typing import Any, Literal, cast

import aiohttp
from google.genai.errors import APIError, ClientError
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    GenerateContentResponse,
    GoogleSearch,
    HarmCategory,
    Part,
    SafetySetting,
    Schema,
    Tool,
)
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_PROMPT, DOMAIN
from .entity import GoogleGenerativeAILLMBaseEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [GoogleGenerativeAIConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class GoogleGenerativeAIConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    GoogleGenerativeAILLMBaseEntity,
):
    """Google Generative AI conversation agent."""

    _attr_supports_streaming = True

    def __init__(self, entry: ConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def _fix_tool_name(self, tool_name: str) -> str:
        """Fix tool name if needed."""
        # The Gemini 2.0+ tokenizer seemingly has a issue with the HassListAddItem tool
        # name. This makes sure when it incorrectly changes the name, that we change it
        # back for HA to call.
        return tool_name if tool_name != "HasListAddItem" else "HassListAddItem"

    async def _fetch_memory_data(self) -> str:
        """Fetch memory data from the API endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:6444/list_memory") as response:
                    if response.status == 200:
                        memory_data = await response.text()
                        return f"\n\nCurrent Memory Data:\n{memory_data}"
                    else:
                        LOGGER.warning("Failed to fetch memory data: HTTP %s", response.status)
                        return ""
        except Exception as err:
            LOGGER.warning("Error fetching memory data: %s", err)
            return ""
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.subentry.data

        # Fetch memory data and append to prompt
        memory_data = await self._fetch_memory_data()
        base_prompt = options.get(CONF_PROMPT, "")
        enhanced_prompt = base_prompt + memory_data

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                enhanced_prompt,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
