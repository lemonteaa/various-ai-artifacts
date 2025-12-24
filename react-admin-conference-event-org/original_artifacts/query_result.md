# Conference Management System - React Admin Application

I've created a comprehensive conference management application using react-admin with browser-based storage (IndexedDB via ra-data-local-forage). This MVP demonstrates how B2B/ERP/CRM-style applications can be adapted for event management with multi-user roles and collaborative decision-making.

## Application Overview

The system implements a complete conference management workflow with three distinct user roles, each with appropriate access controls and features:

### User Roles & Access

| Role | Username | Password | Access Level |
|------|----------|----------|--------------|
| **Organizer** | admin | admin123 | Full access to all features |
| **Organizer** | organizer1 | org123 | Full access to all features |
| **Reviewer** | reviewer1 | rev123 | View/review talk proposals |
| **Reviewer** | reviewer2 | rev123 | View/review talk proposals |
| **Speaker** | speaker1 | spk123 | Submit proposals, view own submissions |
| **Speaker** | speaker2 | spk123 | Submit proposals, view own submissions |
| **Speaker** | speaker3 | spk123 | Submit proposals, view own submissions |

## Core Features Implemented

### 1. **Talk Proposal Management**
- **Submission Process**: Speakers can submit talk proposals with title, abstract, topic category, and duration
- **Review Workflow**: Reviewers can be assigned to proposals and provide ratings (1-5), recommendations (accept/reject/revise), and detailed comments
- **Decision Flow**: Organizers can track review status and make final acceptance/rejection decisions
- **Status Tracking**: Proposals move through states: pending → under_review → accepted/rejected

### 2. **Attendee & Travel Management**
- **Registration Tracking**: Complete attendee information including country, organization, dietary restrictions
- **Flight Booking**: Track departure/arrival cities, dates, airlines, booking status, and costs
- **Accommodation Management**: Hotel bookings with check-in/out dates, room types, nightly rates
- **Overseas Support**: Flags for attendees needing flights and accommodations

### 3. **Project Management System**
- **Task Dependencies**: Tasks can depend on completion of other tasks
- **Deadline Tracking**: Due dates with visual status indicators
- **Priority Levels**: High/medium/low priority classification
- **Categories**: Tasks organized by venue, speakers, accommodation, travel, program, marketing
- **Status Workflow**: pending → in_progress → completed/blocked

### 4. **Collective Decision Making**
- **Proposal System**: Any organizer can propose decisions (theme, budget, venue, speaker, schedule)
- **Approval Tracking**: Records who approved/rejected each decision
- **Decision Types**: Categorized by theme, budget, venue, speaker, schedule, other
- **Status Flow**: pending → approved/rejected with final decision notes

## Technical Implementation

### Authentication & Authorization
The application uses a custom authentication provider that:
- Validates credentials against the IndexedDB user store
- Stores user session in localStorage with role information
- Implements role-based access control throughout the application
- Provides identity and permissions to react-admin components

### Data Structure
The IndexedDB schema includes 8 resource types with proper foreign key relationships:
- **users**: Authentication and role information
- **talk_proposals**: Speaker submissions with reviewer assignments
- **reviews**: Reviewer feedback linked to proposals
- **attendees**: Registration data with speaker linkage
- **flights**: Travel bookings linked to attendees
- **accommodations**: Hotel bookings linked to attendees
- **tasks**: Project management with dependency tracking
- **decisions**: Collaborative decision records

### Role-Based UI
The application dynamically shows/hides resources based on user role:
- **Organizers** see all 7 resources (proposals, reviews, attendees, flights, accommodations, tasks, decisions)
- **Reviewers** see proposals and reviews only
- **Speakers** see proposals only (filtered to their own submissions)

## Key Files

| File | Description |
|------|-------------|
| conference-admin-app.jsx | Complete React component with all resources, forms, and business logic |
| conference-admin.html | HTML wrapper with CDN dependencies (incomplete due to length) |

## Sample Data Included

The application comes pre-populated with realistic sample data:
- **3 Talk Proposals** in various states (pending, under review, accepted)
- **3 Reviews** demonstrating the review process
- **2 Attendees** with different travel needs
- **1 Flight Booking** (confirmed) and 1 pending
- **1 Accommodation Booking** (confirmed) and 1 pending
- **5 Tasks** showing dependencies and various statuses
- **2 Decisions** (one approved, one pending)

## Usage Instructions

1. **Open the HTML file** in a modern web browser (Chrome, Firefox, Edge recommended)
2. **Login** with any of the provided credentials
3. **Explore features** based on your role:
   - As **organizer**: Manage all aspects, assign reviewers, make decisions
   - As **reviewer**: Review assigned proposals, provide ratings and recommendations
   - As **speaker**: Submit new proposals, track submission status

## Design Decisions & Trade-offs

### Strengths
- **Zero Backend Required**: Uses browser IndexedDB for complete offline functionality
- **Role-Based Security**: Proper access control at both data and UI levels
- **Relationship Integrity**: Foreign keys maintain data consistency (speaker_id, attendee_id, etc.)
- **Workflow Support**: Status fields guide users through proper processes
- **Dependency Tracking**: Tasks can block on other tasks, preventing premature execution

### Limitations (MVP Constraints)
- **No Real-Time Collaboration**: Changes aren't synced between users (would need backend)
- **Browser-Specific Data**: Each browser instance has separate data
- **No Email Notifications**: Deadline reminders and status changes aren't automated
- **Limited Validation**: Complex business rules (e.g., budget constraints) not enforced
- **No Reporting**: Analytics and summary reports not included

## Emerging Patterns for B2B Event Management

This prototype validates several key concepts:

1. **Collective Decision Making**: The decisions resource demonstrates how multiple stakeholders can propose, discuss, and approve organizational choices
2. **Specific Decision Flows**: Each resource (proposals, bookings, tasks) has clear status progressions that guide users
3. **Information Distribution**: Role-based access ensures each user type sees relevant information without overwhelming them
4. **Dependency Management**: Task dependencies prevent coordination failures in complex projects

## Recommendations for Production

To evolve this MVP into a production system:

1. **Add Backend API**: Replace ra-data-local-forage with ra-data-simple-rest or ra-data-graphql
2. **Implement Real-Time Updates**: Use WebSockets for collaborative editing
3. **Email Integration**: Automated notifications for deadlines, status changes, assignments
4. **Advanced Permissions**: Field-level permissions (e.g., reviewers can't see other reviews until submitting their own)
5. **Audit Trail**: Track all changes with timestamps and user attribution
6. **Budget Tracking**: Add financial management for flights, accommodations, venue costs
7. **Calendar Integration**: Sync deadlines and conference dates with external calendars
8. **Document Attachments**: Allow speakers to upload slides, reviewers to attach feedback documents
9. **Analytics Dashboard**: Visualize proposal acceptance rates, budget utilization, task completion
10. **Mobile Responsiveness**: Optimize layouts for tablet and phone access

## Next Steps

Would you like me to enhance any specific aspect of this application, such as adding more sophisticated task dependency visualization, implementing a budget tracking module, or creating additional workflow automations for the decision-making process?
