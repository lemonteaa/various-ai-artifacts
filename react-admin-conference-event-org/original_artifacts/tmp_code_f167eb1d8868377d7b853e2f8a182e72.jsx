// filename: conference-admin-app.jsx
// execution: false

import React, { useState } from 'react';
import {
  Admin,
  Resource,
  List,
  Datagrid,
  TextField,
  DateField,
  EmailField,
  ReferenceField,
  Edit,
  Create,
  SimpleForm,
  TextInput,
  DateInput,
  ReferenceInput,
  SelectInput,
  required,
  BooleanField,
  BooleanInput,
  NumberField,
  NumberInput,
  useRecordContext,
  FunctionField,
  ChipField,
  ArrayField,
  SingleFieldList,
  ReferenceArrayInput,
  SelectArrayInput,
  useGetIdentity,
  usePermissions,
  Show,
  SimpleShowLayout,
  RichTextField,
  TabbedShowLayout,
  Tab,
  ReferenceManyField,
  TopToolbar,
  EditButton,
  ShowButton,
  DeleteButton,
  ListButton,
  CreateButton,
  useNotify,
  useRedirect,
  SaveButton,
  Toolbar,
  Button,
  useUpdate,
  useRefresh,
  DateTimeInput,
  Labeled,
} from 'react-admin';
import localForageDataProvider from 'ra-data-local-forage';
import { Card, CardContent, Typography, Box, Chip, Grid } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import PendingIcon from '@mui/icons-material/Pending';
import EventIcon from '@mui/icons-material/Event';
import PeopleIcon from '@mui/icons-material/People';
import FlightIcon from '@mui/icons-material/Flight';
import HotelIcon from '@mui/icons-material/Hotel';
import AssignmentIcon from '@mui/icons-material/Assignment';
import PersonIcon from '@mui/icons-material/Person';

// Default data with users, roles, and sample conference data
const defaultData = {
  users: [
    // Organizers (full access)
    { id: 1, username: 'admin', password: 'admin123', email: 'admin@conference.com', role: 'organizer', name: 'Alice Admin' },
    { id: 2, username: 'organizer1', password: 'org123', email: 'org1@conference.com', role: 'organizer', name: 'Bob Organizer' },
    
    // Reviewers (can review talk proposals)
    { id: 3, username: 'reviewer1', password: 'rev123', email: 'rev1@conference.com', role: 'reviewer', name: 'Carol Reviewer' },
    { id: 4, username: 'reviewer2', password: 'rev123', email: 'rev2@conference.com', role: 'reviewer', name: 'David Reviewer' },
    
    // Speakers (can submit proposals)
    { id: 5, username: 'speaker1', password: 'spk123', email: 'speaker1@conference.com', role: 'speaker', name: 'Eve Speaker' },
    { id: 6, username: 'speaker2', password: 'spk123', email: 'speaker2@conference.com', role: 'speaker', name: 'Frank Speaker' },
    { id: 7, username: 'speaker3', password: 'spk123', email: 'speaker3@conference.com', role: 'speaker', name: 'Grace Speaker' },
  ],
  
  talk_proposals: [
    {
      id: 1,
      title: 'The Future of AI in Healthcare',
      abstract: 'Exploring how artificial intelligence is revolutionizing medical diagnosis and treatment planning.',
      speaker_id: 5,
      submitted_date: '2025-11-15',
      status: 'under_review',
      duration_minutes: 45,
      topic_category: 'AI/ML',
      reviewer_ids: [3, 4],
    },
    {
      id: 2,
      title: 'Sustainable Software Architecture',
      abstract: 'Best practices for building environmentally conscious and energy-efficient software systems.',
      speaker_id: 6,
      submitted_date: '2025-11-20',
      status: 'accepted',
      duration_minutes: 30,
      topic_category: 'Software Engineering',
      reviewer_ids: [3],
      decision_date: '2025-12-01',
      decision_notes: 'Excellent topic, very relevant to current trends.',
    },
    {
      id: 3,
      title: 'Quantum Computing Basics',
      abstract: 'An introduction to quantum computing principles for software developers.',
      speaker_id: 7,
      submitted_date: '2025-11-25',
      status: 'pending',
      duration_minutes: 60,
      topic_category: 'Emerging Tech',
      reviewer_ids: [],
    },
  ],
  
  reviews: [
    {
      id: 1,
      talk_proposal_id: 1,
      reviewer_id: 3,
      rating: 4,
      comments: 'Strong proposal with clear objectives. Would benefit from more technical depth.',
      review_date: '2025-11-22',
      recommendation: 'accept',
    },
    {
      id: 2,
      talk_proposal_id: 1,
      reviewer_id: 4,
      rating: 5,
      comments: 'Excellent topic and well-structured abstract. Highly recommend acceptance.',
      review_date: '2025-11-23',
      recommendation: 'accept',
    },
    {
      id: 3,
      talk_proposal_id: 2,
      reviewer_id: 3,
      rating: 5,
      comments: 'Timely and important topic. Speaker has strong credentials.',
      review_date: '2025-11-28',
      recommendation: 'accept',
    },
  ],
  
  attendees: [
    {
      id: 1,
      name: 'John Attendee',
      email: 'john@company.com',
      country: 'USA',
      organization: 'Tech Corp',
      registration_date: '2025-12-01',
      needs_accommodation: true,
      needs_flight: true,
      dietary_restrictions: 'Vegetarian',
      speaker_id: 5,
    },
    {
      id: 2,
      name: 'Jane International',
      email: 'jane@global.com',
      country: 'UK',
      organization: 'Global Solutions',
      registration_date: '2025-12-03',
      needs_accommodation: true,
      needs_flight: true,
      dietary_restrictions: 'None',
      speaker_id: null,
    },
    {
      id: 3,
      name: 'Mike Local',
      email: 'mike@local.com',
      country: 'USA',
      organization: 'Local Business',
      registration_date: '2025-12-05',
      needs_accommodation: false,
      needs_flight: false,
      dietary_restrictions: 'Gluten-free',
      speaker_id: 6,
    },
  ],
  
  flights: [
    {
      id: 1,
      attendee_id: 1,
      departure_city: 'New York',
      arrival_city: 'San Francisco',
      departure_date: '2026-03-14',
      return_date: '2026-03-18',
      airline: 'United Airlines',
      flight_number: 'UA1234',
      booking_status: 'confirmed',
      cost: 450,
      booking_reference: 'ABC123',
    },
    {
      id: 2,
      attendee_id: 2,
      departure_city: 'London',
      arrival_city: 'San Francisco',
      departure_date: '2026-03-13',
      return_date: '2026-03-19',
      airline: 'British Airways',
      flight_number: 'BA285',
      booking_status: 'pending',
      cost: 850,
      booking_reference: null,
    },
  ],
  
  accommodations: [
    {
      id: 1,
      attendee_id: 1,
      hotel_name: 'Conference Hotel',
      check_in_date: '2026-03-14',
      check_out_date: '2026-03-18',
      room_type: 'Standard',
      booking_status: 'confirmed',
      cost_per_night: 180,
      total_nights: 4,
      booking_reference: 'HTL456',
    },
    {
      id: 2,
      attendee_id: 2,
      hotel_name: 'Conference Hotel',
      check_in_date: '2026-03-13',
      check_out_date: '2026-03-19',
      room_type: 'Deluxe',
      booking_status: 'pending',
      cost_per_night: 220,
      total_nights: 6,
      booking_reference: null,
    },
  ],
  
  tasks: [
    {
      id: 1,
      title: 'Finalize venue contract',
      description: 'Sign the contract with the conference venue and make initial payment.',
      assigned_to_id: 1,
      status: 'completed',
      priority: 'high',
      due_date: '2025-12-01',
      completed_date: '2025-11-28',
      depends_on_task_ids: [],
      category: 'venue',
    },
    {
      id: 2,
      title: 'Send acceptance letters to speakers',
      description: 'Notify all accepted speakers and request confirmation.',
      assigned_to_id: 2,
      status: 'in_progress',
      priority: 'high',
      due_date: '2025-12-20',
      completed_date: null,
      depends_on_task_ids: [],
      category: 'speakers',
    },
    {
      id: 3,
      title: 'Book hotel room blocks',
      description: 'Reserve room blocks at partner hotels for attendees.',
      assigned_to_id: 1,
      status: 'pending',
      priority: 'medium',
      due_date: '2026-01-15',
      completed_date: null,
      depends_on_task_ids: [1],
      category: 'accommodation',
    },
    {
      id: 4,
      title: 'Set up flight booking system',
      description: 'Configure the flight booking portal for international attendees.',
      assigned_to_id: 2,
      status: 'pending',
      priority: 'medium',
      due_date: '2026-01-10',
      completed_date: null,
      depends_on_task_ids: [],
      category: 'travel',
    },
    {
      id: 5,
      title: 'Create conference schedule',
      description: 'Finalize the conference schedule with all accepted talks.',
      assigned_to_id: 1,
      status: 'pending',
      priority: 'high',
      due_date: '2026-01-30',
      completed_date: null,
      depends_on_task_ids: [2],
      category: 'program',
    },
  ],
  
  decisions: [
    {
      id: 1,
      title: 'Conference Theme Selection',
      description: 'Choose the main theme for the 2026 conference.',
      decision_type: 'theme',
      status: 'approved',
      proposed_by_id: 1,
      proposed_date: '2025-10-15',
      decision_date: '2025-10-20',
      approved_by_ids: [1, 2],
      rejected_by_ids: [],
      final_decision: 'Innovation in Technology and Sustainability',
      notes: 'Unanimous approval. Theme resonates with current industry trends.',
    },
    {
      id: 2,
      title: 'Keynote Speaker Budget',
      description: 'Approve budget increase for keynote speaker honorarium.',
      decision_type: 'budget',
      status: 'pending',
      proposed_by_id: 2,
      proposed_date: '2025-12-10',
      decision_date: null,
      approved_by_ids: [2],
      rejected_by_ids: [],
      final_decision: null,
      notes: 'Waiting for financial review.',
    },
  ],
};

// Custom Authentication Provider
const authProvider = {
  login: async ({ username, password }) => {
    const dataProvider = await localForageDataProvider({
      defaultData,
      prefixLocalForageKey: 'conference-app',
    });
    
    const { data: users } = await dataProvider.getList('users', {
      pagination: { page: 1, perPage: 100 },
      sort: { field: 'id', order: 'ASC' },
      filter: {},
    });
    
    const user = users.find(u => u.username === username && u.password === password);
    
    if (user) {
      localStorage.setItem('auth', JSON.stringify({ 
        id: user.id, 
        username: user.username, 
        role: user.role,
        name: user.name,
        email: user.email,
      }));
      return Promise.resolve();
    }
    
    return Promise.reject(new Error('Invalid username or password'));
  },
  
  logout: () => {
    localStorage.removeItem('auth');
    return Promise.resolve();
  },
  
  checkAuth: () => {
    return localStorage.getItem('auth') ? Promise.resolve() : Promise.reject();
  },
  
  checkError: (error) => {
    const status = error.status;
    if (status === 401 || status === 403) {
      localStorage.removeItem('auth');
      return Promise.reject();
    }
    return Promise.resolve();
  },
  
  getIdentity: () => {
    const auth = localStorage.getItem('auth');
    if (auth) {
      const user = JSON.parse(auth);
      return Promise.resolve({
        id: user.id,
        fullName: user.name,
        avatar: null,
      });
    }
    return Promise.reject();
  },
  
  getPermissions: () => {
    const auth = localStorage.getItem('auth');
    if (auth) {
      const user = JSON.parse(auth);
      return Promise.resolve(user.role);
    }
    return Promise.reject();
  },
};

// Initialize data provider
const dataProvider = localForageDataProvider({
  defaultData,
  prefixLocalForageKey: 'conference-app',
});

// Custom Dashboard
const Dashboard = () => {
  const { permissions } = usePermissions();
  const { identity } = useGetIdentity();
  
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Conference Management Dashboard
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" gutterBottom>
        Welcome, {identity?.fullName} ({permissions})
      </Typography>
      
      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <EventIcon sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
                <Typography variant="h6">Talk Proposals</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Manage speaker submissions and review process
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <PeopleIcon sx={{ fontSize: 40, color: 'success.main', mr: 2 }} />
                <Typography variant="h6">Attendees</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Track registrations and attendee information
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <FlightIcon sx={{ fontSize: 40, color: 'info.main', mr: 2 }} />
                <Typography variant="h6">Travel</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Manage flight bookings for overseas attendees
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <AssignmentIcon sx={{ fontSize: 40, color: 'warning.main', mr: 2 }} />
                <Typography variant="h6">Tasks</Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Project management with dependencies and deadlines
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Quick Access
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Use the sidebar menu to navigate to different sections based on your role.
        </Typography>
        
        {permissions === 'organizer' && (
          <Box sx={{ mt: 2 }}>
            <Chip label="Full Access" color="primary" sx={{ mr: 1 }} />
            <Typography variant="body2" sx={{ mt: 1 }}>
              As an organizer, you have access to all features including decisions, task management, and all data.
            </Typography>
          </Box>
        )}
        
        {permissions === 'reviewer' && (
          <Box sx={{ mt: 2 }}>
            <Chip label="Reviewer Access" color="secondary" sx={{ mr: 1 }} />
            <Typography variant="body2" sx={{ mt: 1 }}>
              As a reviewer, you can view and review talk proposals assigned to you.
            </Typography>
          </Box>
        )}
        
        {permissions === 'speaker' && (
          <Box sx={{ mt: 2 }}>
            <Chip label="Speaker Access" color="success" sx={{ mr: 1 }} />
            <Typography variant="body2" sx={{ mt: 1 }}>
              As a speaker, you can submit talk proposals and view your submission status.
            </Typography>
          </Box>
        )}
      </Box>
      
      <Box sx={{ mt: 4, p: 2, bgcolor: 'info.light', borderRadius: 1 }}>
        <Typography variant="body2">
          <strong>Demo Credentials:</strong><br />
          Organizer: admin / admin123<br />
          Reviewer: reviewer1 / rev123<br />
          Speaker: speaker1 / spk123
        </Typography>
      </Box>
    </Box>
  );
};

// Status color helper
const getStatusColor = (status) => {
  const colors = {
    pending: 'default',
    under_review: 'info',
    accepted: 'success',
    rejected: 'error',
    confirmed: 'success',
    completed: 'success',
    in_progress: 'warning',
    approved: 'success',
  };
  return colors[status] || 'default';
};

// Talk Proposals Components
const TalkProposalList = () => {
  const { permissions } = usePermissions();
  const auth = JSON.parse(localStorage.getItem('auth') || '{}');
  
  const filters = permissions === 'speaker' 
    ? { speaker_id: auth.id }
    : {};
  
  return (
    <List filter={filters}>
      <Datagrid rowClick="show">
        <TextField source="title" />
        <ReferenceField source="speaker_id" reference="users" link={false}>
          <TextField source="name" />
        </ReferenceField>
        <DateField source="submitted_date" />
        <FunctionField
          label="Status"
          render={record => (
            <Chip 
              label={record.status.replace('_', ' ')} 
              color={getStatusColor(record.status)}
              size="small"
            />
          )}
        />
        <TextField source="topic_category" />
        <NumberField source="duration_minutes" label="Duration (min)" />
        {permissions === 'organizer' && <EditButton />}
        <ShowButton />
      </Datagrid>
    </List>
  );
};

const TalkProposalShow = () => {
  const { permissions } = usePermissions();
  
  return (
    <Show>
      <TabbedShowLayout>
        <Tab label="Details">
          <TextField source="title" />
          <TextField source="abstract" multiline />
          <ReferenceField source="speaker_id" reference="users" link={false}>
            <TextField source="name" />
          </ReferenceField>
          <DateField source="submitted_date" />
          <FunctionField
            label="Status"
            render={record => (
              <Chip 
                label={record.status.replace('_', ' ')} 
                color={getStatusColor(record.status)}
              />
            )}
          />
          <TextField source="topic_category" />
          <NumberField source="duration_minutes" label="Duration (minutes)" />
          {permissions === 'organizer' && (
            <>
              <DateField source="decision_date" />
              <TextField source="decision_notes" multiline />
            </>
          )}
        </Tab>
        
        {(permissions === 'organizer' || permissions === 'reviewer') && (
          <Tab label="Reviews">
            <ReferenceManyField reference="reviews" target="talk_proposal_id" label="">
              <Datagrid>
                <ReferenceField source="reviewer_id" reference="users" link={false}>
                  <TextField source="name" />
                </ReferenceField>
                <NumberField source="rating" />
                <TextField source="recommendation" />
                <TextField source="comments" />
                <DateField source="review_date" />
              </Datagrid>
            </ReferenceManyField>
          </Tab>
        )}
      </TabbedShowLayout>
    </Show>
  );
};

const TalkProposalEdit = () => (
  <Edit>
    <SimpleForm>
      <TextInput source="title" validate={required()} fullWidth />
      <TextInput source="abstract" multiline rows={4} validate={required()} fullWidth />
      <ReferenceInput source="speaker_id" reference="users" filter={{ role: 'speaker' }}>
        <SelectInput optionText="name" validate={required()} />
      </ReferenceInput>
      <DateInput source="submitted_date" validate={required()} />
      <SelectInput source="status" choices={[
        { id: 'pending', name: 'Pending' },
        { id: 'under_review', name: 'Under Review' },
        { id: 'accepted', name: 'Accepted' },
        { id: 'rejected', name: 'Rejected' },
      ]} validate={required()} />
      <TextInput source="topic_category" validate={required()} />
      <NumberInput source="duration_minutes" validate={required()} />
      <ReferenceArrayInput source="reviewer_ids" reference="users" filter={{ role: 'reviewer' }}>
        <SelectArrayInput optionText="name" />
      </ReferenceArrayInput>
      <DateInput source="decision_date" />
      <TextInput source="decision_notes" multiline rows={3} fullWidth />
    </SimpleForm>
  </Edit>
);

const TalkProposalCreate = () => {
  const auth = JSON.parse(localStorage.getItem('auth') || '{}');
  const { permissions } = usePermissions();
  
  return (
    <Create>
      <SimpleForm defaultValues={{ 
        speaker_id: permissions === 'speaker' ? auth.id : undefined,
        submitted_date: new Date().toISOString().split('T')[0],
        status: 'pending',
      }}>
        <TextInput source="title" validate={required()} fullWidth />
        <TextInput source="abstract" multiline rows={4} validate={required()} fullWidth />
        {permissions === 'organizer' && (
          <ReferenceInput source="speaker_id" reference="users" filter={{ role: 'speaker' }}>
            <SelectInput optionText="name" validate={required()} />
          </ReferenceInput>
        )}
        <TextInput source="topic_category" validate={required()} />
        <NumberInput source="duration_minutes" validate={required()} defaultValue={30} />
      </SimpleForm>
    </Create>
  );
};

// Reviews Components
const ReviewList = () => {
  const auth = JSON.parse(localStorage.getItem('auth') || '{}');
  const { permissions } = usePermissions();
  
  const filters = permissions === 'reviewer' 
    ? { reviewer_id: auth.id }
    : {};
  
  return (
    <List filter={filters}>
      <Datagrid rowClick="edit">
        <ReferenceField source="talk_proposal_id" reference="talk_proposals" link="show">
          <TextField source="title" />
        </ReferenceField>
        <ReferenceField source="reviewer_id" reference="users" link={false}>
          <TextField source="name" />
        </ReferenceField>
        <NumberField source="rating" />
        <FunctionField
          label="Recommendation"
          render={record => (
            <Chip 
              label={record.recommendation} 
              color={record.recommendation === 'accept' ? 'success' : record.recommendation === 'reject' ? 'error' : 'default'}
              size="small"
            />
          )}
        />
        <DateField source="review_date" />
        <EditButton />
      </Datagrid>
    </List>
  );
};

const ReviewEdit = () => (
  <Edit>
    <SimpleForm>
      <ReferenceInput source="talk_proposal_id" reference="talk_proposals">
        <SelectInput optionText="title" validate={required()} disabled />
      </ReferenceInput>
      <NumberInput source="rating" min={1} max={5} validate={required()} />
      <SelectInput source="recommendation" choices={[
        { id: 'accept', name: 'Accept' },
        { id: 'reject', name: 'Reject' },
        { id: 'revise', name: 'Needs Revision' },
      ]} validate={required()} />
      <TextInput source="comments" multiline rows={4} fullWidth validate={required()} />
      <DateInput source="review_date" validate={required()} />
    </SimpleForm>
  </Edit>
);

const ReviewCreate = () => {
  const auth = JSON.parse(localStorage.getItem('auth') || '{}');
  
  return (
    <Create>
      <SimpleForm defaultValues={{ 
        reviewer_id: auth.id,
        review_date: new Date().toISOString().split('T')[0],
      }}>
        <ReferenceInput source="talk_proposal_id" reference="talk_proposals">
          <SelectInput optionText="title" validate={required()} />
        </ReferenceInput>
        <NumberInput source="rating" min={1} max={5} validate={required()} />
        <SelectInput source="recommendation" choices={[
          { id: 'accept', name: 'Accept' },
          { id: 'reject', name: 'Reject' },
          { id: 'revise', name: 'Needs Revision' },
        ]} validate={required()} />
        <TextInput source="comments" multiline rows={4} fullWidth validate={required()} />
      </SimpleForm>
    </Create>
  );
};

// Attendees Components
const AttendeeList = () => (
  <List>
    <Datagrid rowClick="show">
      <TextField source="name" />
      <EmailField source="email" />
      <TextField source="country" />
      <TextField source="organization" />
      <DateField source="registration_date" />
      <BooleanField source="needs_flight" />
      <BooleanField source="needs_accommodation" />
      <EditButton />
      <ShowButton />
    </Datagrid>
  </List>
);

const AttendeeShow = () => (
  <Show>
    <TabbedShowLayout>
      <Tab label="Details">
        <TextField source="name" />
        <EmailField source="email" />
        <TextField source="country" />
        <TextField source="organization" />
        <DateField source="registration_date" />
        <BooleanField source="needs_flight" />
        <BooleanField source="needs_accommodation" />
        <TextField source="dietary_restrictions" />
        <ReferenceField source="speaker_id" reference="users" link={false} emptyText="Not a speaker">
          <TextField source="name" />
        </ReferenceField>
      </Tab>
      
      <Tab label="Travel">
        <ReferenceManyField reference="flights" target="attendee_id" label="Flights">
          <Datagrid>
            <TextField source="departure_city" />
            <TextField source="arrival_city" />
            <DateField source="departure_date" />
            <DateField source="return_date" />
            <TextField source="airline" />
            <FunctionField
              label="Status"
              render={record => (
                <Chip label={record.booking_status} color={getStatusColor(record.booking_status)} size="small" />
              )}
            />
          </Datagrid>
        </ReferenceManyField>
        
        <ReferenceManyField reference="accommodations" target="attendee_id" label="Accommodations">
          <Datagrid>
            <TextField source="hotel_name" />
            <DateField source="check_in_date" />
            <DateField source="check_out_date" />
            <TextField source="room_type" />
            <NumberField source="total_nights" />
            <FunctionField
              label="Status"
              render={record => (
                <Chip label={record.booking_status} color={getStatusColor(record.booking_status)} size="small" />
              )}
            />
          </Datagrid>
        </ReferenceManyField>
      </Tab>
    </TabbedShowLayout>
  </Show>
);

const AttendeeEdit = () => (
  <Edit>
    <SimpleForm>
      <TextInput source="name" validate={required()} />
      <TextInput source="email" type="email" validate={required()} />
      <TextInput source="country" validate={required()} />
      <TextInput source="organization" />
      <DateInput source="registration_date" validate={required()} />
      <BooleanInput source="needs_flight" />
      <BooleanInput source="needs_accommodation" />
      <TextInput source="dietary_restrictions" />
      <ReferenceInput source="speaker_id" reference="users" filter={{ role: 'speaker' }}>
        <SelectInput optionText="name" />
      </ReferenceInput>
    </SimpleForm>
  </Edit>
);

const AttendeeCreate = () => (
  <Create>
    <SimpleForm defaultValues={{ 
      registration_date: new Date().toISOString().split('T')[0],
      needs_flight: false,
      needs_accommodation: false,
    }}>
      <TextInput source="name" validate={required()} />
      <TextInput source="email" type="email" validate={required()} />
      <TextInput source="country" validate={required()} />
      <TextInput source="organization" />
      <BooleanInput source="needs_flight" />
      <BooleanInput source="needs_accommodation" />
      <TextInput source="dietary_restrictions" />
      <ReferenceInput source="speaker_id" reference="users" filter={{ role: 'speaker' }}>
        <SelectInput optionText="name" />
      </ReferenceInput>
    </SimpleForm>
  </Create>
);

// Flights Components
const FlightList = () => (
  <List>
    <Datagrid rowClick="edit">
      <ReferenceField source="attendee_id" reference="attendees" link="show">
        <TextField source="name" />
      </ReferenceField>
      <TextField source="departure_city" />
      <TextField source="arrival_city" />
      <DateField source="departure_date" />
      <DateField source="return_date" />
      <TextField source="airline" />
      <FunctionField
        label="Status"
        render={record => (
          <Chip label={record.booking_status} color={getStatusColor(record.booking_status)} size="small" />
        )}
      />
      <NumberField source="cost" options={{ style: 'currency', currency: 'USD' }} />
      <EditButton />
    </Datagrid>
  </List>
);

const FlightEdit = () => (
  <Edit>
    <SimpleForm>
      <ReferenceInput source="attendee_id" reference="attendees">
        <SelectInput optionText="name" validate={required()} />
      </ReferenceInput>
      <TextInput source="departure_city" validate={required()} />
      <TextInput source="arrival_city" validate={required()} />
      <DateInput source="departure_date" validate={required()} />
      <DateInput source="return_date" validate={required()} />
      <TextInput source="airline" validate={required()} />
      <TextInput source="flight_number" />
      <SelectInput source="booking_status" choices={[
        { id: 'pending', name: 'Pending' },
        { id: 'confirmed', name: 'Confirmed' },
        { id: 'cancelled', name: 'Cancelled' },
      ]} validate={required()} />
      <NumberInput source="cost" validate={required()} />
      <TextInput source="booking_reference" />
    </SimpleForm>
  </Edit>
);

const FlightCreate = () => (
  <Create>
    <SimpleForm defaultValues={{ booking_status: 'pending' }}>
      <ReferenceInput source="attendee_id" reference="attendees">
        <SelectInput optionText="name" validate={required()} />
      </ReferenceInput>
      <TextInput source="departure_city" validate={required()} />
      <TextInput source="arrival_city" validate={required()} />
      <DateInput source="departure_date" validate={required()} />
      <DateInput source="return_date" validate={required()} />
      <TextInput source="airline" validate={required()} />
      <TextInput source="flight_number" />
      <NumberInput source="cost" validate={required()} />
      <TextInput source="booking_reference" />
    </SimpleForm>
  </Create>
);

// Accommodations Components
const AccommodationList = () => (
  <List>
    <Datagrid rowClick="edit">
      <ReferenceField source="attendee_id" reference="attendees" link="show">
        <TextField source="name" />
      </ReferenceField>
      <TextField source="hotel_name" />
      <DateField source="check_in_date" />
      <DateField source="check_out_date" />
      <TextField source="room_type" />
      <NumberField source="total_nights" />
      <FunctionField
        label="Status"
        render={record => (
          <Chip label={record.booking_status} color={getStatusColor(record.booking_status)} size="small" />
        )}
      />
      <FunctionField
        label="Total Cost"
        render={record => `$${(record.cost_per_night * record.total_nights).toFixed(2)}`}
      />
      <EditButton />
    </Datagrid>
  </List>
);

const AccommodationEdit = () => (
  <Edit>
    <SimpleForm>
      <ReferenceInput source="attendee_id" reference="attendees">
        <SelectInput optionText="name" validate={required()} />
      </ReferenceInput>
      <TextInput source="hotel_name" validate={required()} />
      <DateInput source="check_in_date" validate={required()} />
      <DateInput source="check_out_date" validate={required()} />
      <TextInput source="room_type" validate={required()} />
      <SelectInput source="booking_status" choices={[
        { id: 'pending', name: 'Pending' },
        { id: 'confirmed', name: 'Confirmed' },
        { id: 'cancelled', name: 'Cancelled' },
      ]} validate={required()} />
      <NumberInput source="cost_per_night" validate={required()} />
      <NumberInput source="total_nights" validate={required()} />
      <TextInput source="booking_reference" />
    </SimpleForm>
  </Edit>
);

const AccommodationCreate = () => (
  <Create>
    <SimpleForm defaultValues={{ booking_status: 'pending' }}>
      <ReferenceInput source="attendee_id" reference="attendees">
        <SelectInput optionText="name" validate={required()} />
      </ReferenceInput>
      <TextInput source="hotel_name" validate={required()} />
      <DateInput source="check_in_date" validate={required()} />
      <DateInput source="check_out_date" validate={required()} />
      <TextInput source="room_type" validate={required()} />
      <NumberInput source="cost_per_night" validate={required()} />
      <NumberInput source="total_nights" validate={required()} />
      <TextInput source="booking_reference" />
    </SimpleForm>
  </Create>
);

// Tasks Components
const TaskList = () => (
  <List sort={{ field: 'due_date', order: 'ASC' }}>
    <Datagrid rowClick="edit">
      <TextField source="title" />
      <ReferenceField source="assigned_to_id" reference="users" link={false}>
        <TextField source="name" />
      </ReferenceField>
      <FunctionField
        label="Status"
        render={record => (
          <Chip label={record.status.replace('_', ' ')} color={getStatusColor(record.status)} size="small" />
        )}
      />
      <FunctionField
        label="Priority"
        render={record => (
          <Chip 
            label={record.priority} 
            color={record.priority === 'high' ? 'error' : record.priority === 'medium' ? 'warning' : 'default'}
            size="small"
          />
        )}
      />
      <DateField source="due_date" />
      <TextField source="category" />
      <EditButton />
    </Datagrid>
  </List>
);

const TaskEdit = () => (
  <Edit>
    <SimpleForm>
      <TextInput source="title" validate={required()} fullWidth />
      <TextInput source="description" multiline rows={3} fullWidth />
      <ReferenceInput source="assigned_to_id" reference="users" filter={{ role: 'organizer' }}>
        <SelectInput optionText="name" validate={required()} />
      </ReferenceInput>
      <SelectInput source="status" choices={[
        { id: 'pending', name: 'Pending' },
        { id: 'in_progress', name: 'In Progress' },
        { id: 'completed', name: 'Completed' },
        { id: 'blocked', name: 'Blocked' },
      ]} validate={required()} />
      <SelectInput source="priority" choices={[
        { id: 'low', name: 'Low' },
        { id: 'medium', name: 'Medium' },
        { id: 'high', name: 'High' },
      ]} validate={required()} />
      <DateInput source="due_date" validate={required()} />
      <DateInput source="completed_date" />
      <SelectInput source="category" choices={[
        { id: 'venue', name: 'Venue' },
        { id: 'speakers', name: 'Speakers' },
        { id: 'accommodation', name: 'Accommodation' },
        { id: 'travel', name: 'Travel' },
        { id: 'program', name: 'Program' },
        { id: 'marketing', name: 'Marketing' },
        { id: 'other', name: 'Other' },
      ]} validate={required()} />
      <ReferenceArrayInput source="depends_on_task_ids" reference="tasks">
        <SelectArrayInput optionText="title" />
      </ReferenceArrayInput>
    </SimpleForm>
  </Edit>
);

const TaskCreate = () => (
  <Create>
    <SimpleForm defaultValues={{ status: 'pending', priority: 'medium' }}>
      <TextInput source="title" validate={required()} fullWidth />
      <TextInput source="description" multiline rows={3} fullWidth />
      <ReferenceInput source="assigned_to_id" reference="users" filter={{ role: 'organizer' }}>
        <SelectInput optionText="name" validate={required()} />
      </ReferenceInput>
      <SelectInput source="priority" choices={[
        { id: 'low', name: 'Low' },
        { id: 'medium', name: 'Medium' },
        { id: 'high', name: 'High' },
      ]} validate={required()} />
      <DateInput source="due_date" validate={required()} />
      <SelectInput source="category" choices={[
        { id: 'venue', name: 'Venue' },
        { id: 'speakers', name: 'Speakers' },
        { id: 'accommodation', name: 'Accommodation' },
        { id: 'travel', name: 'Travel' },
        { id: 'program', name: 'Program' },
        { id: 'marketing', name: 'Marketing' },
        { id: 'other', name: 'Other' },
      ]} validate={required()} />
      <ReferenceArrayInput source="depends_on_task_ids" reference="tasks">
        <SelectArrayInput optionText="title" />
      </ReferenceArrayInput>
    </SimpleForm>
  </Create>
);

// Decisions Components
const DecisionList = () => (
  <List>
    <Datagrid rowClick="show">
      <TextField source="title" />
      <TextField source="decision_type" />
      <FunctionField
        label="Status"
        render={record => (
          <Chip label={record.status} color={getStatusColor(record.status)} size="small" />
        )}
      />
      <ReferenceField source="proposed_by_id" reference="users" link={false}>
        <TextField source="name" />
      </ReferenceField>
      <DateField source="proposed_date" />
      <DateField source="decision_date" />
      <ShowButton />
      <EditButton />
    </Datagrid>
  </List>
);

const DecisionShow = () => (
  <Show>
    <SimpleShowLayout>
      <TextField source="title" />
      <TextField source="description" />
      <TextField source="decision_type" />
      <FunctionField
        label="Status"
        render={record => (
          <Chip label={record.status} color={getStatusColor(record.status)} />
        )}
      />
      <ReferenceField source="proposed_by_id" reference="users" link={false}>
        <TextField source="name" />
      </ReferenceField>
      <DateField source="proposed_date" />
      <DateField source="decision_date" />
      <TextField source="final_decision" />
      <TextField source="notes" multiline />
      <Labeled label="Approved By">
        <FunctionField
          render={record => {
            if (!record.approved_by_ids || record.approved_by_ids.length === 0) return 'None';
            return record.approved_by_ids.join(', ');
          }}
        />
      </Labeled>
      <Labeled label="Rejected By">
        <FunctionField
          render={record => {
            if (!record.rejected_by_ids || record.rejected_by_ids.length === 0) return 'None';
            return record.rejected_by_ids.join(', ');
          }}
        />
      </Labeled>
    </SimpleShowLayout>
  </Show>
);

const DecisionEdit = () => (
  <Edit>
    <SimpleForm>
      <TextInput source="title" validate={required()} fullWidth />
      <TextInput source="description" multiline rows={3} fullWidth />
      <SelectInput source="decision_type" choices={[
        { id: 'theme', name: 'Theme' },
        { id: 'budget', name: 'Budget' },
        { id: 'venue', name: 'Venue' },
        { id: 'speaker', name: 'Speaker' },
        { id: 'schedule', name: 'Schedule' },
        { id: 'other', name: 'Other' },
      ]} validate={required()} />
      <SelectInput source="status" choices={[
        { id: 'pending', name: 'Pending' },
        { id: 'approved', name: 'Approved' },
        { id: 'rejected', name: 'Rejected' },
      ]} validate={required()} />
      <DateInput source="decision_date" />
      <TextInput source="final_decision" fullWidth />
      <TextInput source="notes" multiline rows={3} fullWidth />
    </SimpleForm>
  </Edit>
);

const DecisionCreate = () => {
  const auth = JSON.parse(localStorage.getItem('auth') || '{}');
  
  return (
    <Create>
      <SimpleForm defaultValues={{ 
        proposed_by_id: auth.id,
        proposed_date: new Date().toISOString().split('T')[0],
        status: 'pending',
        approved_by_ids: [],
        rejected_by_ids: [],
      }}>
        <TextInput source="title" validate={required()} fullWidth />
        <TextInput source="description" multiline rows={3} fullWidth />
        <SelectInput source="decision_type" choices={[
          { id: 'theme', name: 'Theme' },
          { id: 'budget', name: 'Budget' },
          { id: 'venue', name: 'Venue' },
          { id: 'speaker', name: 'Speaker' },
          { id: 'schedule', name: 'Schedule' },
          { id: 'other', name: 'Other' },
        ]} validate={required()} />
        <TextInput source="notes" multiline rows={3} fullWidth />
      </SimpleForm>
    </Create>
  );
};

// Main App Component
const App = () => (
  <Admin 
    dataProvider={dataProvider} 
    authProvider={authProvider}
    dashboard={Dashboard}
  >
    {permissions => (
      <>
        {/* Talk Proposals - All roles can view, speakers can create their own */}
        <Resource
          name="talk_proposals"
          list={TalkProposalList}
          show={TalkProposalShow}
          edit={permissions === 'organizer' ? TalkProposalEdit : undefined}
          create={permissions === 'speaker' || permissions === 'organizer' ? TalkProposalCreate : undefined}
          icon={EventIcon}
          options={{ label: 'Talk Proposals' }}
        />
        
        {/* Reviews - Reviewers and organizers only */}
        {(permissions === 'reviewer' || permissions === 'organizer') && (
          <Resource
            name="reviews"
            list={ReviewList}
            edit={ReviewEdit}
            create={ReviewCreate}
            icon={AssignmentIcon}
            options={{ label: 'Reviews' }}
          />
        )}
        
        {/* Attendees - Organizers only */}
        {permissions === 'organizer' && (
          <Resource
            name="attendees"
            list={AttendeeList}
            show={AttendeeShow}
            edit={AttendeeEdit}
            create={AttendeeCreate}
            icon={PeopleIcon}
            options={{ label: 'Attendees' }}
          />
        )}
        
        {/* Flights - Organizers only */}
        {permissions === 'organizer' && (
          <Resource
            name="flights"
            list={FlightList}
            edit={FlightEdit}
            create={FlightCreate}
            icon={FlightIcon}
            options={{ label: 'Flights' }}
          />
        )}
        
        {/* Accommodations - Organizers only */}
        {permissions === 'organizer' && (
          <Resource
            name="accommodations"
            list={AccommodationList}
            edit={AccommodationEdit}
            create={AccommodationCreate}
            icon={HotelIcon}
            options={{ label: 'Accommodations' }}
          />
        )}
        
        {/* Tasks - Organizers only */}
        {permissions === 'organizer' && (
          <Resource
            name="tasks"
            list={TaskList}
            edit={TaskEdit}
            create={TaskCreate}
            icon={AssignmentIcon}
            options={{ label: 'Tasks' }}
          />
        )}
        
        {/* Decisions - Organizers only */}
        {permissions === 'organizer' && (
          <Resource
            name="decisions"
            list={DecisionList}
            show={DecisionShow}
            edit={DecisionEdit}
            create={DecisionCreate}
            icon={AssignmentIcon}
            options={{ label: 'Decisions' }}
          />
        )}
        
        {/* Users - Reference only, no direct access */}
        <Resource name="users" icon={PersonIcon} />
      </>
    )}
  </Admin>
);

export default App;