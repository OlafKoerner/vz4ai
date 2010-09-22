<?php

namespace Volkszaehler\Model\Proxy;

/**
 * THIS CLASS WAS GENERATED BY THE DOCTRINE ORM. DO NOT EDIT THIS FILE.
 */
class VolkszaehlerModelPropertyProxy extends \Volkszaehler\Model\Property implements \Doctrine\ORM\Proxy\Proxy
{
    private $_entityPersister;
    private $_identifier;
    public $__isInitialized__ = false;
    public function __construct($entityPersister, $identifier)
    {
        $this->_entityPersister = $entityPersister;
        $this->_identifier = $identifier;
    }
    private function _load()
    {
        if (!$this->__isInitialized__ && $this->_entityPersister) {
            $this->__isInitialized__ = true;
            if ($this->_entityPersister->load($this->_identifier, $this) === null) {
                throw new \Doctrine\ORM\EntityNotFoundException();
            }
            unset($this->_entityPersister, $this->_identifier);
        }
    }

    
    public function cast()
    {
        $this->_load();
        return parent::cast();
    }

    public function validate()
    {
        $this->_load();
        return parent::validate();
    }

    public function checkRemove()
    {
        $this->_load();
        return parent::checkRemove();
    }

    public function checkPersist()
    {
        $this->_load();
        return parent::checkPersist();
    }

    public function getKey()
    {
        $this->_load();
        return parent::getKey();
    }

    public function getValue()
    {
        $this->_load();
        return parent::getValue();
    }

    public function getDefinition()
    {
        $this->_load();
        return parent::getDefinition();
    }

    public function setValue($value)
    {
        $this->_load();
        return parent::setValue($value);
    }


    public function __sleep()
    {
        return array('__isInitialized__', 'id', 'key', 'value', 'entity');
    }
}