from __future__ import annotations

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class Meeting(BaseModel):
    """A calendar event, scheduled call, video conference, or sync."""
    title: str = Field(description="Title or subject of the meeting")
    participants: list[str] = Field(default_factory=list, description="List of participant names")
    time: Optional[str] = Field(None, description="Time or date/time string of the meeting")
    platform: Optional[str] = Field(None, description="e.g. Zoom, Google Meet, Microsoft Teams, Slack")
    meeting_link: Optional[str] = Field(None, description="Direct URL/link to join the meeting (e.g. https://meet.google.com/abc-defg-hij)")
    emails: list[str] = Field(default_factory=list, description="Email addresses detected in the meeting context")
    discussion_points: list[str] = Field(default_factory=list, description="Key discussion points or topics mentioned")
    action_items: list[str] = Field(default_factory=list, description="Direct action items or tasks assigned")
    
    # Provenance
    source_frame_ids: list[int] = Field(default_factory=list, description="IDs of frames where this meeting is visible")
    source_timestamps: list[str] = Field(default_factory=list, description="Timestamps of frames where this meeting is visible")
    is_inferred: bool = Field(default=False, description="True if inferred rather than explicitly stated")
    inferred_rationale: Optional[str] = Field(None, description="Explanation for why this was inferred")


class FileActivity(BaseModel):
    """Activity involving a file, document, codebase, or application window."""
    file_name: Optional[str] = Field(None, description="Filename including extension (e.g. main.py)")
    file_path: Optional[str] = Field(None, description="Full or relative directory/file path if visible")
    document_title: Optional[str] = Field(None, description="Title of the document (e.g. slide deck title)")
    application: Optional[str] = Field(None, description="Active application name (e.g. VS Code, Chrome)")
    start_time: Optional[str] = Field(None, description="When the file activity started")
    end_time: Optional[str] = Field(None, description="When the file activity ended")
    estimated_duration: Optional[str] = Field(None, description="Estimated duration (e.g. '5 minutes' or '300 seconds')")

    # Provenance
    source_frame_ids: list[int] = Field(default_factory=list)
    source_timestamps: list[str] = Field(default_factory=list)
    is_inferred: bool = Field(default=False)
    inferred_rationale: Optional[str] = Field(None)


class Appointment(BaseModel):
    """A scheduled point on the calendar, deadline, or reminder."""
    title: str = Field(description="Appointment title or task due")
    date: Optional[str] = Field(None, description="Scheduled date")
    time: Optional[str] = Field(None, description="Scheduled time")
    deadline: Optional[str] = Field(None, description="Associated deadline or due date/time")
    reminder: Optional[str] = Field(None, description="Associated reminder date/time")

    # Provenance
    source_frame_ids: list[int] = Field(default_factory=list)
    source_timestamps: list[str] = Field(default_factory=list)
    is_inferred: bool = Field(default=False)
    inferred_rationale: Optional[str] = Field(None)


class Person(BaseModel):
    """A person referenced on screen."""
    name: str = Field(description="Name or handle of the person")
    email: Optional[str] = Field(None, description="Email address if visible")
    organization: Optional[str] = Field(None, description="Company, team, or institution they belong to")

    # Provenance
    source_frame_ids: list[int] = Field(default_factory=list)
    source_timestamps: list[str] = Field(default_factory=list)
    is_inferred: bool = Field(default=False)
    inferred_rationale: Optional[str] = Field(None)


class Organization(BaseModel):
    """A company, startup, or team name."""
    name: str = Field(description="Name of the organization")
    domain: Optional[str] = Field(None, description="Associated domain or website URL")

    # Provenance
    source_frame_ids: list[int] = Field(default_factory=list)
    source_timestamps: list[str] = Field(default_factory=list)
    is_inferred: bool = Field(default=False)


class Project(BaseModel):
    """A project name, repository, or code initiative."""
    name: str = Field(description="Name of the project")
    description: Optional[str] = Field(None, description="Short summary of goals or context")

    # Provenance
    source_frame_ids: list[int] = Field(default_factory=list)
    source_timestamps: list[str] = Field(default_factory=list)
    is_inferred: bool = Field(default=False)


class URLReference(BaseModel):
    """A website URL observed on screen."""
    url: str = Field(description="Page URL")
    title: Optional[str] = Field(None, description="Page title or browser tab name")

    # Provenance
    source_frame_ids: list[int] = Field(default_factory=list)
    source_timestamps: list[str] = Field(default_factory=list)
    is_inferred: bool = Field(default=False)


class FactEvidence(BaseModel):
    """
    Evidence record detailing verification results for a single extracted fact.
    """
    fact_id: str = Field(description="Unique identifier for the fact")
    fact_type: str = Field(description="Type of entity: e.g. meeting, file_activity, person, organization")
    fact: dict[str, Any] = Field(description="Dictionary representation of the extracted fact")
    source_frame: Optional[int] = Field(None, description="Primary frame ID containing the fact")
    source_timestamp: Optional[str] = Field(None, description="Timestamp of the primary frame")
    evidence_text: Optional[str] = Field(None, description="Matching line or context snippet supporting the fact")
    verification_status: VerificationStatus = Field(description="verified, partially_supported, or unsupported")
    unsupported_fields: list[str] = Field(default_factory=list, description="Fields that were not supported by the raw text")


class StructuredMetadata(BaseModel):
    """
    Consolidated structured metadata extracted by the LLM from one
    or more raw text logs.
    """
    meetings: list[Meeting] = Field(default_factory=list)
    files: list[FileActivity] = Field(default_factory=list)
    appointments: list[Appointment] = Field(default_factory=list)
    
    # Entities
    people: list[Person] = Field(default_factory=list)
    organizations: list[Organization] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    urls: list[URLReference] = Field(default_factory=list)


# For backward compatibility
class ScreenContext(StructuredMetadata):
    """Alias for ScreenContext to support older pipeline stages."""
    activity_summary: Optional[str] = Field(None, description="High-level activity summary")
    frame_id: Optional[int] = None
