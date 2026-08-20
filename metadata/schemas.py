"""
metadata/schemas.py

Strongly typed Pydantic models for Stage 3 Metadata Extraction and Verification.
Preserves strict evidence, provenance, and verification state for all extracted facts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, model_validator


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class Evidence(BaseModel):
    """
    Direct evidence supporting an extracted fact or entity.
    Provides unified compatibility for fact verification records.
    """
    fact_id:             str                      = Field(default="", description="Unique identifier for the fact")
    fact_type:           str                      = Field(default="", description="Type of fact (meeting, file_activity, etc.)")
    fact:                dict[str, Any]           = Field(default_factory=dict, description="Dictionary representation of fact")
    raw_text_record_id:  Optional[Union[str, int]] = Field(default=None, description="Source raw text record ID")
    source_frame:        Optional[int]            = Field(default=None, description="Primary source frame ID")
    source_frame_id:     Optional[int]            = Field(default=None, description="Source frame ID alias")
    source_timestamp:    Optional[str]            = Field(default=None, description="Source capture timestamp")
    timestamp:           Optional[str]            = Field(default=None, description="Capture timestamp alias")
    application:         Optional[str]            = Field(default=None, description="Foreground application")
    window_title:        Optional[str]            = Field(default=None, description="Foreground window title")
    evidence_text:       Optional[str]            = Field(default=None, description="Text snippet grounded in source")
    verification_status: VerificationStatus       = Field(default=VerificationStatus.VERIFIED)
    unsupported_fields:  list[str]                = Field(default_factory=list, description="Fields not supported by text")
    confidence:          Optional[float]          = Field(default=1.0, description="Evidence confidence [0.0-1.0]")

    @model_validator(mode="after")
    def sync_aliases(self) -> "Evidence":
        if self.source_frame is None and self.source_frame_id is not None:
            self.source_frame = self.source_frame_id
        elif self.source_frame_id is None and self.source_frame is not None:
            self.source_frame_id = self.source_frame

        if self.source_timestamp is None and self.timestamp is not None:
            self.source_timestamp = self.timestamp
        elif self.timestamp is None and self.source_timestamp is not None:
            self.timestamp = self.source_timestamp

        if not self.raw_text_record_id and self.source_frame is not None:
            self.raw_text_record_id = f"rt_{self.source_frame}"

        return self

    model_config = {"frozen": False, "extra": "allow"}


# Alias for backward compatibility
FactEvidence = Evidence


class Meeting(BaseModel):
    """A calendar event, scheduled call, video conference, or sync."""
    id:                 Optional[str]     = Field(default=None)
    title:              str               = Field(description="Title or subject of the meeting")
    participants:       list[str]         = Field(default_factory=list, description="List of participant names")
    start_time:         Optional[str]     = Field(default=None, description="Start time or date/time string")
    end_time:           Optional[str]     = Field(default=None, description="End time if available")
    time:               Optional[str]     = Field(default=None, description="Time string (alias for start_time)")
    platform:           Optional[str]     = Field(default=None, description="e.g. Zoom, Google Meet, Microsoft Teams")
    meeting_link:       Optional[str]     = Field(default=None, description="Direct URL/link to join meeting")
    emails:             list[str]         = Field(default_factory=list, description="Email addresses in context")
    discussion_points:  list[str]         = Field(default_factory=list, description="Key discussion topics")
    action_items:       list[str]         = Field(default_factory=list, description="Direct action items")
    confidence:         Optional[float]   = Field(default=1.0, description="Fact confidence [0.0-1.0]")
    evidence:           list[Evidence]    = Field(default_factory=list, description="Grounded evidence items")

    # Provenance
    source_frame_ids:   list[int]         = Field(default_factory=list, description="IDs of source frames")
    source_timestamps:  list[str]         = Field(default_factory=list, description="Timestamps of source frames")
    is_inferred:        bool              = Field(default=False, description="True if inferred rather than explicit")
    inferred_rationale: Optional[str]     = Field(default=None, description="Explanation for inference")

    model_config = {"frozen": False, "extra": "allow"}


class FileActivity(BaseModel):
    """Activity involving a file, document, codebase, or application window."""
    id:                 Optional[str]     = Field(default=None)
    file_name:          Optional[str]     = Field(default=None, description="Filename with extension")
    file_path:          Optional[str]     = Field(default=None, description="Full or relative directory/file path")
    document_title:     Optional[str]     = Field(default=None, description="Document title")
    application:        Optional[str]     = Field(default=None, description="Active application name")
    start_time:         Optional[str]     = Field(default=None, description="When file activity started")
    end_time:           Optional[str]     = Field(default=None, description="When file activity ended")
    estimated_duration: Optional[str]     = Field(default=None, description="Estimated duration (e.g. '5m 30s')")
    confidence:         Optional[float]   = Field(default=1.0, description="Fact confidence [0.0-1.0]")
    evidence:           list[Evidence]    = Field(default_factory=list, description="Grounded evidence items")

    # Provenance
    source_frame_ids:   list[int]         = Field(default_factory=list)
    source_timestamps:  list[str]         = Field(default_factory=list)
    is_inferred:        bool              = Field(default=False)
    inferred_rationale: Optional[str]     = Field(default=None)

    model_config = {"frozen": False, "extra": "allow"}


class Appointment(BaseModel):
    """A scheduled point on the calendar, deadline, or reminder."""
    id:                 Optional[str]     = Field(default=None)
    title:              str               = Field(description="Appointment title or task due")
    date:               Optional[str]     = Field(default=None, description="Scheduled date")
    time:               Optional[str]     = Field(default=None, description="Scheduled time")
    deadline:           Optional[str]     = Field(default=None, description="Associated deadline")
    reminder:           Optional[str]     = Field(default=None, description="Associated reminder")
    confidence:         Optional[float]   = Field(default=1.0, description="Fact confidence [0.0-1.0]")
    evidence:           list[Evidence]    = Field(default_factory=list, description="Grounded evidence items")

    # Provenance
    source_frame_ids:   list[int]         = Field(default_factory=list)
    source_timestamps:  list[str]         = Field(default_factory=list)
    is_inferred:        bool              = Field(default=False)
    inferred_rationale: Optional[str]     = Field(default=None)

    model_config = {"frozen": False, "extra": "allow"}


class Entity(BaseModel):
    """A general named entity (person, organization, project, URL, other)."""
    id:                 Optional[str]     = Field(default=None)
    type:               str               = Field(description="person | organization | project | URL | other")
    name:               str               = Field(description="Name or identifier of the entity")
    first_seen:         Optional[str]     = Field(default=None, description="First timestamp observed")
    last_seen:          Optional[str]     = Field(default=None, description="Last timestamp observed")
    occurrences:        int               = Field(default=1, description="Number of times seen")
    confidence:         Optional[float]   = Field(default=1.0, description="Entity confidence [0.0-1.0]")
    evidence:           list[Evidence]    = Field(default_factory=list, description="Grounded evidence items")

    # Provenance
    source_frame_ids:   list[int]         = Field(default_factory=list)
    source_timestamps:  list[str]         = Field(default_factory=list)
    is_inferred:        bool              = Field(default=False)

    model_config = {"frozen": False, "extra": "allow"}


class Person(BaseModel):
    """A person referenced on screen."""
    id:                 Optional[str]     = Field(default=None)
    name:               str               = Field(description="Name or handle of the person")
    email:              Optional[str]     = Field(default=None, description="Email address")
    organization:       Optional[str]     = Field(default=None, description="Company, team, or institution")
    confidence:         Optional[float]   = Field(default=1.0)
    evidence:           list[Evidence]    = Field(default_factory=list)

    # Provenance
    source_frame_ids:   list[int]         = Field(default_factory=list)
    source_timestamps:  list[str]         = Field(default_factory=list)
    is_inferred:        bool              = Field(default=False)
    inferred_rationale: Optional[str]     = Field(default=None)

    model_config = {"frozen": False, "extra": "allow"}


class Organization(BaseModel):
    """A company, startup, or team name."""
    id:                 Optional[str]     = Field(default=None)
    name:               str               = Field(description="Name of the organization")
    domain:             Optional[str]     = Field(default=None, description="Domain or website URL")
    confidence:         Optional[float]   = Field(default=1.0)
    evidence:           list[Evidence]    = Field(default_factory=list)

    # Provenance
    source_frame_ids:   list[int]         = Field(default_factory=list)
    source_timestamps:  list[str]         = Field(default_factory=list)
    is_inferred:        bool              = Field(default=False)

    model_config = {"frozen": False, "extra": "allow"}


class Project(BaseModel):
    """A project name, repository, or code initiative."""
    id:                 Optional[str]     = Field(default=None)
    name:               str               = Field(description="Name of the project")
    description:        Optional[str]     = Field(default=None, description="Short summary of goals")
    confidence:         Optional[float]   = Field(default=1.0)
    evidence:           list[Evidence]    = Field(default_factory=list)

    # Provenance
    source_frame_ids:   list[int]         = Field(default_factory=list)
    source_timestamps:  list[str]         = Field(default_factory=list)
    is_inferred:        bool              = Field(default=False)

    model_config = {"frozen": False, "extra": "allow"}


class URLReference(BaseModel):
    """A website URL observed on screen."""
    id:                 Optional[str]     = Field(default=None)
    url:                str               = Field(description="Page URL")
    title:              Optional[str]     = Field(default=None, description="Page title or tab name")
    confidence:         Optional[float]   = Field(default=1.0)
    evidence:           list[Evidence]    = Field(default_factory=list)

    # Provenance
    source_frame_ids:   list[int]         = Field(default_factory=list)
    source_timestamps:  list[str]         = Field(default_factory=list)
    is_inferred:        bool              = Field(default=False)

    model_config = {"frozen": False, "extra": "allow"}


class StructuredMetadata(BaseModel):
    """
    Consolidated structured metadata extracted from raw text records.
    """
    meetings:      list[Meeting]      = Field(default_factory=list)
    files:         list[FileActivity] = Field(default_factory=list)
    appointments:  list[Appointment]  = Field(default_factory=list)
    entities:      list[Entity]       = Field(default_factory=list)

    # Granular entity lists
    people:        list[Person]       = Field(default_factory=list)
    organizations: list[Organization] = Field(default_factory=list)
    projects:      list[Project]      = Field(default_factory=list)
    urls:          list[URLReference] = Field(default_factory=list)

    model_config = {"frozen": False, "extra": "allow"}


class VerificationResult(BaseModel):
    """Result of running deterministic verification on StructuredMetadata."""
    verified_metadata:    StructuredMetadata = Field(description="Cleaned metadata container")
    evidences:            list[Evidence]     = Field(default_factory=list)
    total_facts:          int                = Field(default=0)
    verified_count:       int                = Field(default=0)
    partially_supported:  int                = Field(default=0)
    unsupported_count:    int                = Field(default=0)
    overall_confidence:   float              = Field(default=1.0)

    model_config = {"frozen": False, "extra": "allow"}


# For backward compatibility
class ScreenContext(StructuredMetadata):
    activity_summary: Optional[str] = Field(default=None, description="High-level activity summary")
    frame_id:         Optional[int] = None
