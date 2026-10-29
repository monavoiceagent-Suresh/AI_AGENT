ALTER TABLE agents DROP CONSTRAINT agents_workflow_check;
ALTER TABLE agents ADD CONSTRAINT agents_workflow_check
    CHECK (workflow IN ('general', 'lead_qualification', 'faq', 'appointment', 'order'));
